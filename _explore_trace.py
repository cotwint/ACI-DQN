"""Quick exploration of Alibaba PAI trace schema and distributions."""
import pandas as pd, numpy as np

# Column names from .header files
task_cols = ['job_name','task_name','inst_num','status','start_time','end_time','plan_cpu','plan_mem','plan_gpu','gpu_type']
job_cols = ['job_name','inst_id','user','status','start_time','end_time']
inst_cols = ['job_name','task_name','inst_name','worker_name','inst_id','status','start_time','end_time','machine']
group_cols = ['inst_id','user','gpu_type_spec','group','workload']
sensor_cols = ['job_name','task_name','worker_name','inst_id','machine','gpu_name','cpu_usage','gpu_wrk_util',
               'avg_mem','max_mem','avg_gpu_wrk_mem','max_gpu_wrk_mem','read','write','read_count','write_count']
machine_cols = ['machine','gpu_type','cap_cpu','cap_mem','cap_gpu']

# --- Task table ---
print("=" * 60)
print("1. TASK TABLE (first 500k rows)")
print("=" * 60)
t = pd.read_csv('trace_dataset/pai_task_table.csv', names=task_cols, nrows=500000)
print(f"Rows: {len(t)}")
print(f"\nStatus counts:\n{t['status'].value_counts()}")
print(f"\nplan_cpu:\n{t['plan_cpu'].describe()}")
print(f"\nplan_gpu:\n{t['plan_gpu'].describe()}")
print(f"\nplan_mem:\n{t['plan_mem'].describe()}")
print(f"\ngpu_type counts:\n{t['gpu_type'].value_counts()}")

# Duration
ok = t['start_time'].notna() & t['end_time'].notna()
dur = t.loc[ok, 'end_time'] - t.loc[ok, 'start_time']
print(f"\nDuration (end-start), unit likely seconds:\n{dur.describe()}")
print(f"Duration quantiles: 50%={dur.quantile(0.5):.0f}, 80%={dur.quantile(0.8):.0f}, 95%={dur.quantile(0.95):.0f}, 99%={dur.quantile(0.99):.0f}")

# Time range
print(f"\nstart_time range: {t['start_time'].min():.0f} .. {t['start_time'].max():.0f}")
print(f"end_time range: {t['end_time'].min():.0f} .. {t['end_time'].max():.0f}")

# --- Group tag table ---
print("\n" + "=" * 60)
print("2. GROUP TAG TABLE (first 500k rows)")
print("=" * 60)
g = pd.read_csv('trace_dataset/pai_group_tag_table.csv', names=group_cols, nrows=500000)
print(f"Rows: {len(g)}")
print(f"\ngroup counts:\n{g['group'].value_counts()}")
print(f"\nworkload counts (top 20):\n{g['workload'].value_counts().head(20)}")
print(f"\ngpu_type_spec counts:\n{g['gpu_type_spec'].value_counts()}")
print(f"\nuser counts (top 10):\n{g['user'].value_counts().head(10)}")

# --- Machine spec ---
print("\n" + "=" * 60)
print("3. MACHINE SPEC TABLE")
print("=" * 60)
ms = pd.read_csv('trace_dataset/pai_machine_spec.csv', names=machine_cols)
print(f"Rows: {len(ms)}")
print(f"\ngpu_type counts:\n{ms['gpu_type'].value_counts()}")
print(f"\ncap_cpu:\n{ms['cap_cpu'].describe()}")
print(f"\ncap_mem:\n{ms['cap_mem'].describe()}")
print(f"\ncap_gpu:\n{ms['cap_gpu'].describe()}")
print(f"\nFirst 5 rows:\n{ms.head()}")

# --- Sensor table sample ---
print("\n" + "=" * 60)
print("4. SENSOR TABLE (first 200k rows)")
print("=" * 60)
s = pd.read_csv('trace_dataset/pai_sensor_table.csv', names=sensor_cols, nrows=200000)
print(f"Rows: {len(s)}")
print(f"\ncpu_usage:\n{s['cpu_usage'].describe()}")
print(f"\ngpu_wrk_util:\n{s['gpu_wrk_util'].describe()}")
print(f"\navg_mem:\n{s['avg_mem'].describe()}")
print(f"\ngpu_name counts:\n{s['gpu_name'].value_counts()}")

# --- Job table sample ---
print("\n" + "=" * 60)
print("5. JOB TABLE (first 200k rows)")
print("=" * 60)
j = pd.read_csv('trace_dataset/pai_job_table.csv', names=job_cols, nrows=200000)
print(f"Rows: {len(j)}")
print(f"\nstatus counts:\n{j['status'].value_counts()}")
print(f"\nuser counts (top 10):\n{j['user'].value_counts().head(10)}")

# --- Instance table sample ---
print("\n" + "=" * 60)
print("6. INSTANCE TABLE (first 200k rows)")
print("=" * 60)
ins = pd.read_csv('trace_dataset/pai_instance_table.csv', names=inst_cols, nrows=200000)
print(f"Rows: {len(ins)}")
print(f"\nstatus counts:\n{ins['status'].value_counts()}")
print(f"\nUnique machines: {ins['machine'].nunique()}")
print(f"Unique job_names: {ins['job_name'].nunique()}")
print(f"Unique inst_names: {ins['inst_name'].nunique()}")

print("\nDone.")
