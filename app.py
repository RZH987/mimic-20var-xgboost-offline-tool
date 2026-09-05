"""离线 Tkinter 工具：最终冻结的20变量 XGBoost 模型。"""
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import joblib
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "selected_reduced_XGBoost.joblib"
THRESHOLD = 0.23393866
MODEL_VERSION = "Reduced XGBoost 20-variable frozen model (MIMIC development 5-fold OOF threshold)"

FEATURES = [
    ("creatinine_last_baseline_ratio", "末次/基线肌酐比", "比值", 0.05, 20),
    ("creatinine_slope_48h", "肌酐斜率（0–48 h）", "mg/dL/h", -5, 5),
    ("aki_stage_at_48h", "48 h AKI分期", "0–3级", 0, 3),
    ("bun_last", "末次BUN", "mg/dL", 1, 300),
    ("urine_rate_24h_min", "24 h最低尿量率", "mL/kg/h", 0, 20),
    ("sofa", "首日SOFA总分", "分", 0, 24),
    ("creatinine_sd", "肌酐标准差", "mg/dL", 0, 10),
    ("glucose_max", "最高血糖", "mg/dL", 20, 1000),
    ("bun_creatinine_ratio_slope_48h", "BUN/肌酐比斜率", "比值/h", -50, 50),
    ("urine_rate_change_48h", "尿量率变化（48 h）", "mL/kg/h", -20, 20),
    ("urine_rate_0_24h", "0–24 h平均尿量率", "mL/kg/h", 0, 20),
    ("bun_creatinine_ratio_last", "末次BUN/肌酐比", "比值", 0, 500),
    ("lactate_max", "最高乳酸", "mmol/L", 0.1, 30),
    ("ptt_max", "最高PTT", "秒", 10, 200),
    ("pt_max", "最高PT", "秒", 5, 150),
    ("heart_rate_max", "最高心率", "次/分", 20, 300),
    ("sodium_max", "最高钠", "mEq/L", 100, 250),
    ("fluid_balance_48h_ml", "48 h液体平衡", "mL", -20000, 50000),
    ("bun_slope_48h", "BUN斜率（0–48 h）", "mg/dL/h", -10, 10),
    ("urine_rate_6h_min", "6 h最低尿量率", "mL/kg/h", 0, 20),
]
FEATURE_NAMES = [x[0] for x in FEATURES]

MODEL = joblib.load(MODEL_PATH)


def predict_row(values):
    """对一行字典预测；返回概率、分类、缺失和范围提示。"""
    row = {}
    missing = []
    warnings = []
    for name, label, unit, low, high in FEATURES:
        text = str(values.get(name, "")).strip()
        if text == "":
            row[name] = np.nan
            missing.append(f"{label}（{name}）")
            continue
        try:
            value = float(text)
            row[name] = value
            if value < low or value > high:
                warnings.append(f"{label}={value:g} {unit} 超出提示范围 [{low:g}, {high:g}]，请核对单位/录入。")
        except ValueError:
            row[name] = np.nan
            missing.append(f"{label}（非数值，按缺失处理）")
    data = pd.DataFrame([row], columns=FEATURE_NAMES)
    probability = float(MODEL.predict_proba(data)[0, 1])
    predicted_class = int(probability >= THRESHOLD)
    return probability, predicted_class, missing, warnings


def main():
    root = tk.Tk()
    root.title("MIMIC 最终20变量 XGBoost 离线工具")
    root.geometry("980x760")

    ttk.Label(root, text="最终冻结20变量 XGBoost（离线）", font=("Microsoft YaHei", 16, "bold")).pack(pady=8)
    ttk.Label(root, text=f"模型版本：{MODEL_VERSION}\nYouden阈值（开发集5折OOF锁定）：{THRESHOLD:.8f}（界面显示 {THRESHOLD:.3f}）").pack()

    frame = ttk.Frame(root)
    frame.pack(fill="both", expand=True, padx=12, pady=8)
    entries = {}
    for i, (name, label, unit, low, high) in enumerate(FEATURES):
        col = 0 if i < 10 else 3
        row = i if i < 10 else i - 10
        ttk.Label(frame, text=label).grid(row=row, column=col, sticky="e", padx=4, pady=3)
        entry = ttk.Entry(frame, width=16)
        entry.grid(row=row, column=col + 1, sticky="w", padx=4, pady=3)
        ttk.Label(frame, text=unit).grid(row=row, column=col + 2, sticky="w", padx=2, pady=3)
        entries[name] = entry

    result_text = tk.StringVar(value="请输入20个变量后点击“预测”。")
    ttk.Label(root, textvariable=result_text, justify="left", foreground="#174a7e").pack(anchor="w", padx=16, pady=5)
    prompt_text = tk.StringVar(value="")
    ttk.Label(root, textvariable=prompt_text, justify="left", foreground="#9a4d00", wraplength=940).pack(anchor="w", padx=16, pady=4)

    last_result = {"probability": None, "predicted_class": None, "missing": [], "warnings": []}

    def do_predict():
        values = {name: entry.get() for name, entry in entries.items()}
        probability, predicted_class, missing, warnings = predict_row(values)
        last_result.update(probability=probability, predicted_class=predicted_class, missing=missing, warnings=warnings)
        label = "阳性（达到阈值）" if predicted_class else "阴性（低于阈值）"
        result_text.set(f"预测概率：{probability:.8f}\n阈值分类：{label}（{predicted_class}）\n模型版本：{MODEL_VERSION}")
        prompts = []
        if missing:
            prompts.append("缺失提示：" + "、".join(missing) + "。模型将按训练期中位数插补，不会重新拟合插补器。")
        if warnings:
            prompts.append("范围/单位提示：" + "；".join(warnings))
        prompt_text.set("\n".join(prompts))

    def clear_all():
        for entry in entries.values():
            entry.delete(0, tk.END)
        result_text.set("请输入20个变量后点击“预测”。")
        prompt_text.set("")
        last_result.update(probability=None, predicted_class=None, missing=[], warnings=[])

    def save_result():
        if last_result["probability"] is None:
            messagebox.showwarning("提示", "请先完成预测。")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV文件", "*.csv")], initialfile="xgboost_prediction.csv")
        if not path:
            return
        values = {name: entries[name].get() for name in FEATURE_NAMES}
        values.update({"predicted_probability": last_result["probability"], "predicted_class": last_result["predicted_class"], "threshold": THRESHOLD, "model_version": MODEL_VERSION, "missing_prompt": "；".join(last_result["missing"]), "range_unit_prompt": "；".join(last_result["warnings"])})
        pd.DataFrame([values]).to_csv(path, index=False, encoding="utf-8-sig")
        messagebox.showinfo("已保存", f"结果已保存到：\n{path}")

    buttons = ttk.Frame(root)
    buttons.pack(pady=8)
    ttk.Button(buttons, text="预测", command=do_predict).pack(side="left", padx=6)
    ttk.Button(buttons, text="清空", command=clear_all).pack(side="left", padx=6)
    ttk.Button(buttons, text="保存结果", command=save_result).pack(side="left", padx=6)
    ttk.Label(root, text="说明：本工具只调用冻结模型和训练期预处理，不联网、不训练、不调参、不重新选择阈值。范围提示不自动修改输入值。", foreground="#555555").pack(pady=4)
    root.mainloop()


if __name__ == "__main__":
    main()
