"""数据审计脚本：定期检查目标值、特征、断层数据质量"""
from db_utils import DBUtils
import pandas as pd
import numpy as np

def audit_data_quality():
    db = DBUtils(config_path="config.ini")
    # 1. 审计t_prediction_parameters（核心参数表）
    df = pd.read_sql("SELECT * FROM t_prediction_parameters", db.engine)
    print("=== 核心参数表审计结果 ===")
    # 目标值审计
    target_features = ["gas_emission_q", "drilling_cuttings_s", "gas_emission_velocity_q"]
    for target in target_features:
        var = df[target].var()
        print(f"{target}：方差={var:.6f}（建议>1e-6）")
    # 2. 审计t_geo_fault（断层表）
    fault_df = pd.read_sql("SELECT id, name, workface_id FROM t_geo_fault", db.engine)
    print("\n=== 断层表审计结果 ===")
    invalid_faults = 0
    for _, row in fault_df.iterrows():
        points_df = pd.read_sql(f"SELECT * FROM t_coal_point WHERE geofault_id={row['id']}", db.engine)
        if len(points_df) < 2:
            invalid_faults += 1
            print(f"断层{row['id']}（{row['name']}）：无效（组成点{len(points_df)}个）")
    print(f"无效断层占比：{invalid_faults/len(fault_df):.2%}（建议<50%）")

if __name__ == "__main__":
    audit_data_quality()