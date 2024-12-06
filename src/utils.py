from tqdm import tqdm
import pickle
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import math
import os

def run_kfold_trainonly(
	train_features: np.ndarray,
    train_labels: np.ndarray,
    n_folds: int,
    model: object,
    model_params: dict,
    seed=42
):
    kf = KFold(n_splits=n_folds)
    if seed:
        kf = KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed
        )
    oof_preds = []
    oof_labels = []
    train_preds = np.zeros((len(train_labels)))
    feat_importances = np.zeros((len(train_features[0])))
    feat_importance_dict = {}

    bar = tqdm(total=n_folds)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_features, train_labels)):
        # print(train_labels.shape, train_idx.shape, val_idx.shape)
        x_train, x_val = train_features[train_idx], train_features[val_idx]
        y_train, y_val = train_labels[train_idx], train_labels[val_idx]
        reg = model(**model_params)
        reg.fit(x_train, y_train)

        # Prediction on train data
        preds = reg.predict(x_train)
        train_preds[train_idx] += preds

        # Prediction on val data
        preds = reg.predict(x_val)
        oof_preds.extend(preds.tolist())
        oof_labels.extend(y_val.tolist())

        # Feature importance
        feat_importances += reg.feature_importances_
        feat_importance_dict[fold] = reg.feature_importances_

        bar.update()

    train_preds /= (n_folds - 1)
    feat_importances /= n_folds

    return train_preds, oof_preds, oof_labels, feat_importances, feat_importance_dict

def calculate_metric(predictions: np.ndarray, labels: np.ndarray):
    # Calculate the absolute errors
    mae = np.mean(abs(predictions - labels))

    # calculate mean absolute errors
    mape = np.mean(np.abs((predictions - labels) / labels)) * 100

    # squared root of mean squared error
    rmse = math.sqrt(mean_squared_error(labels, predictions))

    rsqr = r2_score(labels, predictions)
    return mae, mape, rmse, rsqr

def run_kfold(
	train_features: np.ndarray, 
  train_labels: np.ndarray, 
  test_features: np.ndarray,
  test_labels: np.ndarray,
  n_folds: int, 
  model: object, 
  model_params: dict,
  output_path: str,
  name: str='rf_model', 
  seed=42
):  
    kf = KFold(n_splits=n_folds)
    if seed:
        kf = KFold(
            n_splits=n_folds, 
            shuffle=True,
            random_state=seed
        )
    oof_preds = []
    train_preds = []
    test_preds = []

    train_metrics = {}
    val_metrics = {}
    test_metrics = {}

    feat_importances = np.zeros((len(train_features[0])))

    bar = tqdm(total=n_folds)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_features)):
        x_train, x_val = train_features[train_idx], train_features[val_idx]
        y_train, y_val = train_labels[train_idx], train_labels[val_idx]
        reg = model(**model_params)
        reg.fit(x_train, y_train)
        
        # Prediction on train data
        preds = reg.predict(x_train)
        train_preds.append(preds)

        # save metrics
        mae, mape, rmse, rsqr = calculate_metric(preds, y_train)
        train_metrics[f"mae_fold{fold}"] = mae
        train_metrics[f"mape_fold{fold}"] = mape
        train_metrics[f"rmse_fold{fold}"] = rmse
        train_metrics[f"rsqr_fold{fold}"] = rsqr
        
        # Prediction on val data
        preds = reg.predict(x_val)
        oof_preds.append(preds)

        # save metrics
        mae, mape, rmse, rsqr = calculate_metric(preds, y_val)
        val_metrics[f"mae_fold{fold}"] = mae
        val_metrics[f"mape_fold{fold}"] = mape
        val_metrics[f"rmse_fold{fold}"] = rmse
        val_metrics[f"rsqr_fold{fold}"] = rsqr
        
        # Prediction on test data
        preds = reg.predict(test_features)
        test_preds.append(preds)

        # save metrics
        mae, mape, rmse, rsqr = calculate_metric(preds, test_labels)
        test_metrics[f"mae_fold{fold}"] = mae
        test_metrics[f"mape_fold{fold}"] = mape
        test_metrics[f"rmse_fold{fold}"] = rmse
        test_metrics[f"rsqr_fold{fold}"] = rsqr

        # Feature importance
        if hasattr(reg, 'feature_importances_'):
            feat_importances += reg.feature_importances_
        else:
            feat_importances += reg.coef_

        pickle.dump(
            reg, 
            open(os.path.join(output_path, f"{name}_{seed}_fold-{fold + 1}.pkl"), "wb")
        )
        bar.update()
        
    feat_importances /= n_folds

    return train_preds, test_preds, oof_preds, feat_importances, train_metrics, val_metrics, test_metrics

  
def run_kfold_eval(
	features: np.ndarray, 
  labels: np.ndarray, 
  n_folds: int, 
  model: object, 
  model_params: dict | None,
  output_path: str,
  name: str='rf_model', 
  seed=42
):  
    kf = KFold(n_splits=n_folds)
    if seed:
        kf = KFold(
            n_splits=n_folds, 
            shuffle=True,
            random_state=seed
        )
    oof_preds = []
    train_preds = []

    train_metrics = {}
    val_metrics = {}

    feat_importances = np.zeros(features.shape[1])

    bar = tqdm(total=n_folds)
    for fold, (train_idx, val_idx) in enumerate(kf.split(features)):

        if model_params != None:
            x_train, x_val = features[train_idx], features[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            reg = model(**model_params)
        else:
            x_train, x_val = features.iloc[train_idx], features.iloc[val_idx]
            y_train, y_val = labels[train_idx], labels[val_idx]
            reg = model
        reg.fit(x_train, y_train)
        
        # Prediction on train data
        preds = reg.predict(x_train)
        train_preds.append(preds)

        # save metrics
        mae, mape, rmse, rsqr = calculate_metric(preds, y_train)
        train_metrics[f"mae_fold{fold}"] = mae
        train_metrics[f"mape_fold{fold}"] = mape
        train_metrics[f"rmse_fold{fold}"] = rmse
        train_metrics[f"rsqr_fold{fold}"] = rsqr
        
        # Prediction on val data
        preds = reg.predict(x_val)
        oof_preds.append(preds)

        # save metrics
        mae, mape, rmse, rsqr = calculate_metric(preds, y_val)
        val_metrics[f"mae_fold{fold}"] = mae
        val_metrics[f"mape_fold{fold}"] = mape
        val_metrics[f"rmse_fold{fold}"] = rmse
        val_metrics[f"rsqr_fold{fold}"] = rsqr

        # Feature importance
        if hasattr(reg, 'feature_importances_'):
            feat_importances += reg.feature_importances_
        elif hasattr(reg, 'coef_'):
            feat_importances += reg.coef_

        pickle.dump(
            reg, 
            open(os.path.join(output_path, f"{name}_{seed}_fold-{fold + 1}.pkl"), "wb")
        )
        bar.update()
        
    feat_importances /= n_folds

    return train_preds, oof_preds, feat_importances, train_metrics, val_metrics

def feature_selection(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    features: list,
    models: list,
    seeds: list,
    model_params: list,
    n_folds: int,
    features_threshold=None,
    topk_features=None
):
    total_features_importance = np.zeros((len(features)))
    features_importance_dict = {}
    for i, model in enumerate(models):
        for j, seed in enumerate(seeds):
            _, _, _, feat_imp, feat_imp_dict = run_kfold_trainonly(
                train_features,
                train_labels,
                n_folds,
                model,
                model_params[i],
                seed=seed
            )
            total_features_importance += feat_imp
            features_importance_dict[f'model{i}_seed{j}'] = feat_imp_dict

    total_features_importance /= (len(models) * len(seeds))
    total_features = [
        [features[i], total_features_importance[i]] for i in range(len(features))
    ]
    total_features.sort(key=lambda x: x[1], reverse=True)

    # plt.barh([el[0] for el in total_features], [el[1] for el in total_features])
    # plt.yticks(fontsize='xx-small')
    # plt.show()

    if topk_features:
        # return [el[0] for el in total_features[:topk_features]]
        # print(total_features)
        return total_features[:topk_features], features_importance_dict
    elif features_threshold:
        return_features = []
        for f in total_features:
            if f[1] < features_threshold:
                break
            return_features.append(f[0])
        return return_features, features_importance_dict
    else:
        raise ValueError("Either topk_features or features_threshold must be specified")
