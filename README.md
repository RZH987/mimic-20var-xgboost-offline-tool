# 最终20变量 XGBoost 离线工具

运行 `E:\文章复现\MIMIC-IV\paper\投稿\本地部署\app.py`（Anaconda Python）。工具使用同目录的冻结 `selected_reduced_XGBoost.joblib`，不联网、不训练、不调参，也不重新拟合插补器或阈值。

界面中的20个变量必须按 `selected_reduced_variables.csv` 的顺序和单位录入。空值会显示缺失提示，并调用模型内已经拟合好的训练期中位数插补；超出提示范围的数值只产生单位/异常范围提示，不自动修改输入。预测结果包含概率、固定阈值0.23393866（显示0.234）、分类、提示和模型版本，可保存为 UTF-8-SIG CSV。

`preprocessor_parameters.csv` 是从冻结模型导出的只读审计表（训练期中位数、均值和标准差），实际预测仍直接调用 joblib 内的预处理器。

固定变量顺序：

`creatinine_last_baseline_ratio, creatinine_slope_48h, aki_stage_at_48h, bun_last, urine_rate_24h_min, sofa, creatinine_sd, glucose_max, bun_creatinine_ratio_slope_48h, urine_rate_change_48h, urine_rate_0_24h, bun_creatinine_ratio_last, lactate_max, ptt_max, pt_max, heart_rate_max, sodium_max, fluid_balance_48h_ml, bun_slope_48h, urine_rate_6h_min`

验证脚本 `validate_sample.py` 从已有 MIMIC 测试集预测结果抽取样本，重新调用同一冻结工具函数，核对概率（绝对误差容限 `1e-6`）和阈值分类。
