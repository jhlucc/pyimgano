import os
import numpy as np
from typing import Dict, List, Optional
import joblib
from tqdm import tqdm
import json
from datetime import datetime

# 导入三个检测器类
from IsolationForest import IsolationForestStructureDetector
from SSIM_struct import MultiTemplatePopupStructDetector
from LOF import LOFStructureAnomalyDetector


class EnsembleVotingDetector:
    """
    集成投票检测器
    整合三个模型的预测结果：
    - IsolationForest (结构异常检测)
    - SSIM Template (模板匹配+弹窗检测)
    - LOF (局部异常因子)

    规则：
    1) SSIM 优先：如果 SSIM 判定“正常”，直接输出正常并返回。
    2) 否则进入加权多数(>=50%)：对当次有效模型做权重归一后投票。
    """

    def __init__(self,
                 iforest_model_path: str = "iforest_structure_detector.pkl",
                 ssim_model_path: str = "multi_template_popup_detector.pkl",
                 lof_model_path: str = "lof_structure_detector.pkl",
                 weights: Optional[Dict[str, float]] = None):
        self.model_paths = {
            'iforest': iforest_model_path,
            'ssim': ssim_model_path,
            'lof': lof_model_path
        }

        # 默认权重（简单投票时各模型权重相等）
        self.weights = weights or {
            'iforest': 1.0,
            'ssim': 1.0,
            'lof': 1.0
        }

        self.models: Dict[str, Optional[object]] = {}
        self.is_loaded = False
        self._load_all_models()

    def _load_all_models(self):
        print("=== 开始加载集成投票模型 ===")
        try:
            # IsolationForest
            print("\n1. 加载IsolationForest模型...")
            if os.path.exists(self.model_paths['iforest']):
                self.models['iforest'] = IsolationForestStructureDetector()
                self.models['iforest'].load(self.model_paths['iforest'])
                print(f"   ✓ IsolationForest模型加载成功: {self.model_paths['iforest']}")
            else:
                print(f"   ✗ IsolationForest模型文件不存在: {self.model_paths['iforest']}")
                self.models['iforest'] = None

            # SSIM
            print("\n2. 加载SSIM模板检测模型...")
            if os.path.exists(self.model_paths['ssim']):
                self.models['ssim'] = MultiTemplatePopupStructDetector()
                self.models['ssim'].load(self.model_paths['ssim'])
                print(f"   ✓ SSIM模型加载成功: {self.model_paths['ssim']}")
            else:
                print(f"   ✗ SSIM模型文件不存在: {self.model_paths['ssim']}")
                self.models['ssim'] = None

            # LOF
            print("\n3. 加载LOF模型...")
            if os.path.exists(self.model_paths['lof']):
                self.models['lof'] = LOFStructureAnomalyDetector()
                self.models['lof'].load(self.model_paths['lof'])
                print(f"   ✓ LOF模型加载成功: {self.model_paths['lof']}")
            else:
                print(f"   ✗ LOF模型文件不存在: {self.model_paths['lof']}")
                self.models['lof'] = None

            loaded_models = [n for n, m in self.models.items() if m is not None]
            if not loaded_models:
                raise ValueError("没有成功加载任何模型！")

            print(f"\n=== 模型加载完成 ===")
            print(f"成功加载 {len(loaded_models)}/{len(self.models)} 个模型: {', '.join(loaded_models)}")

            # 若有模型缺失，对“已加载模型”做一次权重归一
            if len(loaded_models) < len(self.models):
                print("\n调整模型权重（仅对已加载模型归一）...")
                total_weight = sum(self.weights.get(name, 1.0) for name in loaded_models)
                for name in self.models:
                    if self.models[name] is None:
                        self.weights[name] = 0.0
                        print(f"   {name} 权重设为 0")
                if total_weight > 0:
                    for name in loaded_models:
                        self.weights[name] = self.weights.get(name, 1.0) / total_weight
                        print(f"   {name} 权重调整为 {self.weights[name]:.3f}")

            self.is_loaded = True

        except Exception as e:
            print(f"\n加载模型时出错: {e}")
            raise

    def predict(self, image_path: str, return_details: bool = True) -> Dict:
        if not self.is_loaded:
            raise ValueError("模型未加载！")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        results = {
            'image': os.path.basename(image_path),
            'model_predictions': {},
            'votes': {'normal': 0, 'anomaly': 0},  # 仅用于展示
            'weighted_score': 0.0,                 # 支持“正常”的权重占比 [0,1]
            'is_normal': None,
            'prediction': None,
            'confidence': 0.0,
            'anomaly_types': []
        }

        # ============== 0) 先跑 SSIM（优先规则） ==============
        ssim_name = 'ssim'
        ssim_ok = False
        ssim_is_normal = None
        if self.models.get(ssim_name) is not None:
            try:
                ssim_raw = self.models[ssim_name].predict(image_path)
                ssim_pred = {
                    'is_normal': ssim_raw['is_normal'],
                    'prediction': ssim_raw['prediction'],
                    'confidence': float(ssim_raw.get('confidence', 0.5)),
                    'raw_result': ssim_raw if return_details else None
                }
                results['model_predictions'][ssim_name] = ssim_pred
                ssim_ok = True
                ssim_is_normal = ssim_raw['is_normal']

                if ssim_is_normal:
                    # SSIM 判正常 → 直接返回正常
                    results['votes']['normal'] = 1
                    results['weighted_score'] = 1.0
                    results['is_normal'] = True
                    results['prediction'] = '正常'
                    results['confidence'] = ssim_pred['confidence']  # 直接用 SSIM 的置信度
                    results['summary'] = {
                        'total_models_loaded': len([m for m in self.models.values() if m is not None]),
                        'total_models_valid': 1,
                        'votes_normal': results['votes']['normal'],
                        'votes_anomaly': results['votes']['anomaly'],
                        'weighted_score': results['weighted_score'],
                        'decision_rule': 'SSIM优先：SSIM判正常直接返回'
                    }
                    return results
                else:
                    # SSIM 判异常，记录异常类型（用于后续展示）
                    results['votes']['anomaly'] += 1
                    if ssim_raw.get('has_popup'):
                        results['anomaly_types'].append(f"检测到{ssim_raw.get('popup_count', 0)}个弹窗")
                    else:
                        results['anomaly_types'].append("界面结构异常")

            except Exception as e:
                print(f"模型 SSIM 预测出错: {e}")
                results['model_predictions'][ssim_name] = {'error': str(e), 'is_normal': None}

        # ============== 1) 跑其余模型（以及可能的 SSIM 异常结果）===============
        # 把需要参与加权投票的模型列出来（如果 SSIM 成功且为异常，它也会参与投票）
        order = ['iforest', 'lof']  # 其余模型；SSIM 的异常结果已在results中
        for model_name in order:
            model = self.models.get(model_name)
            if model is None:
                continue
            try:
                pred_raw = model.predict(image_path)
                pred_std = {
                    'is_normal': pred_raw['is_normal'],
                    'prediction': pred_raw['prediction'],
                    'confidence': float(pred_raw.get('confidence', 0.5)),
                    'raw_result': pred_raw if return_details else None
                }
                results['model_predictions'][model_name] = pred_std

                if pred_raw['is_normal']:
                    results['votes']['normal'] += 1
                else:
                    results['votes']['anomaly'] += 1
                    # 收集异常类型
                    if model_name == 'iforest':
                        results['anomaly_types'].append(f"结构异常({pred_raw.get('anomaly_level', '未知')})")
                    elif model_name == 'lof':
                        results['anomaly_types'].append(f"局部异常({pred_raw.get('anomaly_level', '未知')})")

            except Exception as e:
                print(f"模型 {model_name} 预测出错: {e}")
                results['model_predictions'][model_name] = {'error': str(e), 'is_normal': None}

        # ============== 2) 有效模型 + 权重 做加权多数决策 ==============
        valid = [(name, pred) for name, pred in results['model_predictions'].items()
                 if pred.get('is_normal') is not None and 'error' not in pred]
        if not valid:
            raise ValueError("没有模型成功预测！")

        total_w = sum(self.weights.get(name, 1.0) for name, _ in valid)
        if total_w <= 0:
            eff_w = {name: 1.0 / len(valid) for name, _ in valid}
        else:
            eff_w = {name: self.weights.get(name, 1.0) / total_w for name, _ in valid}

        normal_w = sum(eff_w[name] for name, pred in valid if pred['is_normal'])
        results['weighted_score'] = float(normal_w)
        results['is_normal'] = normal_w >= 0.5
        results['prediction'] = '正常' if results['is_normal'] else '异常'

        # 综合置信度：0.6*一致性 + 0.4*（按权重的置信度一致度）
        consistency = abs(normal_w - 0.5) * 2  # 0~1
        avg_conf = 0.0
        for name, pred in valid:
            c = float(pred.get('confidence', 0.5))
            agree = (pred['is_normal'] == results['is_normal'])
            avg_conf += eff_w[name] * (c if agree else (1 - c))
        results['confidence'] = float(0.6 * consistency + 0.4 * avg_conf)

        results['summary'] = {
            'total_models_loaded': len([m for m in self.models.values() if m is not None]),
            'total_models_valid': len(valid),
            'votes_normal': results['votes']['normal'],
            'votes_anomaly': results['votes']['anomaly'],
            'weighted_score': results['weighted_score'],
            'decision_rule': 'SSIM优先；若SSIM异常或不可用→加权多数(>=50%)'
        }

        # 去重异常类型
        results['anomaly_types'] = list(set(results['anomaly_types']))
        return results

    def _convert_numpy_types(self, obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                              np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(v) for v in obj)
        else:
            return obj

    def batch_predict(self, test_folder: str, save_results: bool = True) -> List[Dict]:
        if not os.path.exists(test_folder):
            raise FileNotFoundError(f"测试目录不存在: {test_folder}")

        results = []
        anomalies = []
        model_agreement_stats = {'full': 0, 'majority': 0, 'split': 0}

        jpg_files = [f for f in os.listdir(test_folder) if f.lower().endswith('.jpg')]
        if not jpg_files:
            print(f"在 {test_folder} 中没有找到jpg文件")
            return results

        print(f"\n=== 开始批量检测 ===")
        print(f"测试目录: {test_folder}")
        print(f"图片数量: {len(jpg_files)}")
        print(f"使用模型: {', '.join([name for name, model in self.models.items() if model is not None])}")

        for filename in tqdm(jpg_files, desc="检测进度"):
            try:
                img_path = os.path.join(test_folder, filename)
                result = self.predict(img_path, return_details=False)
                results.append(result)

                if not result['is_normal']:
                    anomalies.append(result)

                nw = result.get('weighted_score', 0.0)
                if nw >= 0.999 or nw <= 0.001:
                    model_agreement_stats['full'] += 1
                elif nw > 0.5:
                    model_agreement_stats['majority'] += 1
                else:
                    model_agreement_stats['split'] += 1

            except Exception as e:
                print(f"\n检测 {filename} 出错: {e}")

        n_total = len(results)
        n_normal = sum(1 for r in results if r['is_normal'])
        n_anomaly = n_total - n_normal

        print(f"\n=== 检测结果统计 ===")
        print(f"总数: {n_total}")
        print(f"正常: {n_normal} ({(n_normal / n_total * 100) if n_total else 0:.1f}%)")
        print(f"异常: {n_anomaly} ({(n_anomaly / n_total * 100) if n_total else 0:.1f}%)")

        print(f"\n=== 模型一致性统计（基于加权得分）===")
        if n_total > 0:
            print(f"完全一致: {model_agreement_stats['full']} ({model_agreement_stats['full'] / n_total * 100:.1f}%)")
            print(f"多数一致: {model_agreement_stats['majority']} ({model_agreement_stats['majority'] / n_total * 100:.1f}%)")
            print(f"意见分歧: {model_agreement_stats['split']} ({model_agreement_stats['split'] / n_total * 100:.1f}%)")

        model_stats = {name: {'normal': 0, 'anomaly': 0, 'error': 0}
                       for name in self.models if self.models[name] is not None}
        for result in results:
            for model_name, pred in result['model_predictions'].items():
                if 'error' in pred:
                    model_stats[model_name]['error'] += 1
                elif pred.get('is_normal') is True:
                    model_stats[model_name]['normal'] += 1
                elif pred.get('is_normal') is False:
                    model_stats[model_name]['anomaly'] += 1

        print(f"\n=== 各模型检测统计（未加权，仅参考）===")
        for model_name, stats in model_stats.items():
            total_pred = stats['normal'] + stats['anomaly']
            if total_pred > 0 or stats['error'] > 0:
                print(f"\n{model_name}:")
                if total_pred > 0:
                    print(f"  正常: {stats['normal']} ({stats['normal'] / total_pred * 100:.1f}%)")
                    print(f"  异常: {stats['anomaly']} ({stats['anomaly'] / total_pred * 100:.1f}%)")
                if stats['error'] > 0:
                    print(f"  错误: {stats['error']}")

        anomaly_type_count = {}
        for anomaly in anomalies:
            for atype in anomaly.get('anomaly_types', []):
                anomaly_type_count[atype] = anomaly_type_count.get(atype, 0) + 1

        if anomaly_type_count and n_anomaly > 0:
            print(f"\n=== 异常类型分布 ===")
            for atype, count in sorted(anomaly_type_count.items(), key=lambda x: x[1], reverse=True):
                print(f"{atype}: {count} ({count / n_anomaly * 100:.1f}%)")

        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = f"ensemble_results_{timestamp}.json"
            save_data = {
                'test_folder': test_folder,
                'timestamp': timestamp,
                'statistics': {
                    'total': n_total,
                    'normal': n_normal,
                    'anomaly': n_anomaly,
                    'model_agreement': model_agreement_stats,
                    'model_stats': model_stats,
                    'anomaly_types': anomaly_type_count
                },
                'model_config': {
                    'models': list(self.models.keys()),
                    'weights': self.weights,
                    'model_paths': self.model_paths
                },
                'results': results
            }
            save_data = self._convert_numpy_types(save_data)
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存到: {result_file}")

        return results

    def get_model_info(self) -> Dict:
        info = {
            'loaded_models': [],
            'model_paths': self.model_paths,
            'weights': self.weights,
            'is_loaded': self.is_loaded
        }
        for name, model in self.models.items():
            if model is not None:
                model_info = {
                    'name': name,
                    'type': type(model).__name__,
                    'is_trained': getattr(model, 'is_trained', False)
                }
                if name == 'iforest' and hasattr(model, 'iforest'):
                    model_info['n_estimators'] = getattr(model.iforest, 'n_estimators', None)
                    model_info['contamination'] = getattr(model.iforest, 'contamination', None)
                elif name == 'ssim':
                    model_info['num_templates'] = len(getattr(model, 'templates', {}))
                    model_info['structure_threshold'] = getattr(model, 'structure_threshold', None)
                elif name == 'lof' and hasattr(model, 'lof'):
                    model_info['n_neighbors'] = getattr(model.lof, 'n_neighbors', None)
                    model_info['contamination'] = getattr(model.lof, 'contamination', None)
                info['loaded_models'].append(model_info)
        return info


# 使用示例
if __name__ == "__main__":
    ensemble = EnsembleVotingDetector(
        iforest_model_path="iforest_structure_detector.pkl",
        ssim_model_path="multi_template_popup_detector.pkl",
        lof_model_path="lof_structure_detector.pkl",
        weights={'iforest': 1.0, 'ssim': 1.0, 'lof': 1.0}
    )

    print("\n=== 模型信息 ===")
    info = ensemble.get_model_info()
    print(f"已加载模型: {len(info['loaded_models'])}")
    for model in info['loaded_models']:
        print(f"  - {model['name']} ({model['type']})")

    try:
        test_image = "/data/temp11/正常/青春版/0dd011082dad48e6a782c9b4ea6ee5ea.jpg"
        if os.path.exists(test_image):
            print(f"\n=== 测试单张图片: {test_image} ===")
            result = ensemble.predict(test_image)

            print(f"\n最终预测: {result['prediction']}")
            print(f"置信度: {result['confidence']:.3f}")
            print(f"投票(未加权): 正常={result['votes']['normal']}, 异常={result['votes']['anomaly']}")
            print(f"加权得分(支持正常的权重占比): {result['weighted_score']:.3f}")

            print("\n各模型预测:")
            for model_name, pred in result['model_predictions'].items():
                if 'error' not in pred:
                    print(f"  - {model_name}: {pred['prediction']} (置信度: {pred['confidence']:.3f})")

            if result['anomaly_types']:
                print(f"\n异常类型: {', '.join(result['anomaly_types'])}")
    except Exception as e:
        print(f"测试单张图片出错: {e}")

    # 批量测试按需开启
    # try:
    #     test_folder = "/data/temp10/select1"
    #     if os.path.exists(test_folder):
    #         print(f"\n=== 批量测试: {test_folder} ===")
    #         _ = ensemble.batch_predict(test_folder, save_results=True)
    # except Exception as e:
    #     print(f"批量测试出错: {e}")
