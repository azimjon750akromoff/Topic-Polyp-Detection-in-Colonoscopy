# import json
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path
# from matplotlib.backends.backend_pdf import PdfPages
#
# run = Path("runs/detect/real_polyp")
# json_file = run/"per_epoch_metrics.json"
# out_dir = run/"report"
# out_dir.mkdir(exist_ok=True)
#
# with open(json_file) as f:
#     data = json.load(f)
#
# rows=[]
# for item in data:
#     ep=item["epoch"]
#     rows.append({
#         "epoch":ep,
#         "train_precision":item["train"]["precision"],
#         "train_recall":item["train"]["recall"],
#         "train_f1":item["train"]["f1"],
#         "train_acc":item["train"]["accuracy"],
#         "val_precision":item["val"]["precision"],
#         "val_recall":item["val"]["recall"],
#         "val_f1":item["val"]["f1"],
#         "val_acc":item["val"]["accuracy"],
#     })
#
# df=pd.DataFrame(rows)
# df.to_csv(out_dir/"metrics.csv",index=False)
#
#
# def plot_curve(col,name):
#     plt.figure(figsize=(6,4))
#     plt.plot(df["epoch"],df[col],marker="o")
#     plt.title(name)
#     plt.xlabel("Epoch")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(out_dir/f"{col}.png")
#     plt.close()
#
# for c in df.columns:
#     if c!="epoch":
#         plot_curve(c,c)
#
# # PDF
# with PdfPages(out_dir/"training_report.pdf") as pdf:
#     for png in sorted(out_dir.glob("*.png")):
#         img = plt.imread(png)
#         plt.figure(figsize=(8,6))
#         plt.imshow(img)
#         plt.axis("off")
#         pdf.savefig()
#         plt.close()
#
# print("PDF saved:", out_dir/"training_report.pdf")
