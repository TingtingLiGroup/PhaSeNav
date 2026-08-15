import pandas as pd
import json
from Bio import Entrez
import numpy as np

# 设置Entrez邮箱（必需）
Entrez.email = "569123894@qq.com"

# preprocess for baoyq's data 
def convert_csv_to_fasta(csv_file, fasta_file):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 打开 FASTA 文件进行写入
    with open(fasta_file, 'w') as f:
        for index, row in df.iterrows():
            # 提取 Seqname 和 sequence
            seqname = row['Seqname']
            sequence = row['sequence']

            # 写入 FASTA 格式
            f.write(f">{seqname}\n")
            f.write(f"{sequence}\n")

# preprocess for wxx's data 
def fetch_protein_sequence(protein_id):
    """
    从UniProt获取蛋白质序列
    """
    try:
        # 使用Entrez从UniProt获取记录
        handle = Entrez.efetch(db="protein", id=protein_id, rettype="fasta", retmode="text")
        record = handle.read()
        # 提取序列
        sequence = record.split('\n', 1)[1].replace('\n', '')
        return sequence.strip()
    except Exception as e:
        print(f"Error fetching sequence for {protein_id}: {e}")
        return None


def get_fasta_via_seqid(csv_file, json_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 存储唯一蛋白质ID和它们的标签
    unique_proteins = {}
    entry_weight = set(df['Entry'].tolist())
    
    # 遍历每一行
    for _, row in df.iterrows():
        # 分割Members列中的蛋白质ID
        protein_ids = row['Members'].split()
        subgroup_code = row['Subgroup_code']
        
        # 对每个蛋白质ID处理
        for protein_id in protein_ids:
            if protein_id not in unique_proteins:
                unique_proteins[protein_id] = {str(i): 0 for i in range(18)}
            
            # 对应子组设为1
            unique_proteins[protein_id][str(subgroup_code)] = 1
    
    print(entry_weight)
    # 最终结果列表
    result = []
    
    # 获取每个蛋白质的序列和其他信息
    for protein_id, labels in unique_proteins.items():
        # 获取序列
        sequence = fetch_protein_sequence(protein_id)

        weight = 1 if protein_id in entry_weight else 0
        
        if sequence:
            # 构建条目
            entry = {
                "protein_id": protein_id,
                "sequence": sequence,
                "labels": labels,
                "weight": weight
            }
            print(entry)
            result.append(entry)
        
    
    # 保存为JSON文件
    with open(json_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"总共处理了 {len(result)} 个唯一蛋白质")
    return result

def get_fasta_via_seqid_specific_subgroups(csv_file, json_file, target_subgroups=[4, 8, 15]):
    """
    获取特定子组的蛋白质 FASTA 信息
    
    参数:
    - csv_file: 输入的 CSV 文件路径
    - json_file: 输出的 JSON 文件路径
    - target_subgroups: 目标子组列表，默认为 [4, 8, 15]
    
    返回:
    处理后的蛋白质信息列表
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 存储唯一蛋白质ID和它们的标签
    unique_proteins = {}
    entry_weight = set(df['Entry'].tolist())
    
    # 将目标子组转换为字符串类型（如果输入是整数）
    target_subgroups = [str(subgroup) for subgroup in target_subgroups]
    
    # 遍历每一行，只处理目标子组
    for _, row in df.iterrows():
        # 检查子组是否在目标子组中
        if str(row['Subgroup_code']) in target_subgroups:
            # 分割Members列中的蛋白质ID
            protein_ids = row['Members'].split()
            subgroup_code = row['Subgroup_code']
            
            # 对每个蛋白质ID处理
            for protein_id in protein_ids:
                if protein_id not in unique_proteins:
                    unique_proteins[protein_id] = {str(i): 0 for i in range(18)}
                
                # 对应子组设为1
                unique_proteins[protein_id][str(subgroup_code)] = 1
    
    # 打印权重信息
    print("Entry weight:", entry_weight)
    
    # 最终结果列表
    result = []
    
    # 获取每个蛋白质的序列和其他信息
    for protein_id, labels in unique_proteins.items():
        # 获取序列
        sequence = fetch_protein_sequence(protein_id)
        weight = 1 if protein_id in entry_weight else 0
        
        if sequence:
            # 构建条目
            entry = {
                "protein_id": protein_id,
                "sequence": sequence,
                "labels": labels,
                "weight": weight
            }
            print(entry)
            result.append(entry)
    
    # 保存为JSON文件
    with open(json_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"总共处理了 {len(result)} 个唯一蛋白质")
    return result

# 获得protgps中的全部protein_id
import json
import time
from Bio import ExPASy
from Bio import SwissProt
from urllib.error import HTTPError
from tqdm import tqdm  # 进度条库

def fetch_uniprot_id_batch(entry_names, delay=0.5, max_retries=1):
    """批量获取UniProt ID（带进度显示）"""
    results = {}
    
    # 添加进度条
    with tqdm(total=len(entry_names), desc="Fetching UniProt IDs", unit="entry") as pbar:
        for name in entry_names:
            retries = 0
            while retries < max_retries:
                try:
                    handle = ExPASy.get_sprot_raw(name)
                    record = SwissProt.read(handle)
                    handle.close()
                    results[name] = record.accessions[0]
                    time.sleep(delay)
                    
                    # 更新进度信息（当前处理的entry名称）
                    pbar.set_postfix_str(f"Current: {name[:20]}...")
                    break
                except HTTPError as e:
                    if e.code == 404:
                        results[name] = None
                        break
                    retries += 1
                    time.sleep(2 ** retries)
                except Exception as e:
                    print(f"\nError fetching {name}: {str(e)}")
                    retries += 1
                    time.sleep(1)
            else:
                results[name] = None
                
            # 更新进度条
            pbar.update(1)
    
    return results

def add_protein_id_to_json_optimized(input_file, output_file):
    """带进度显示的JSON处理"""
    print("Loading JSON data...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # 显示总条目数
    entry_names = [item["entry"] for item in data if "entry" in item]
    print(f"Total entries to process: {len(entry_names)}")
    
    # 批量获取（带进度条）
    id_mapping = fetch_uniprot_id_batch(entry_names)
    
    print("\nUpdating JSON records...")
    # 添加更新进度条
    for item in tqdm(data, desc="Updating JSON", unit="record"):
        if "entry" in item:
            item["protein_id"] = id_mapping.get(item["entry"])
    
    print("Saving results...")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"\nDone! Results saved to {output_file}")



if __name__ == "__main__":
    # 替换为你的文件路径
    csv_file = 'Joined_PS_Clusters_basedata_18clusters.csv'  # 输入的 csv 文件路径
    fasta_file = 'sequence.fasta'          # 输出的 FASTA 文件路径
    json_file = 'sae_18clusters.json'
    # json_analyze_file = 'sae_18clusters_subgroups_15.json'
    # 使用示例
    # convert_csv_to_fasta(csv_file, fasta_file)
    # get_fasta_via_seqid(csv_file, json_file)
    # get_fasta_via_seqid_specific_subgroups(csv_file, json_analyze_file,target_subgroups=[15])
    # print(f"FASTA file '{fasta_file}' created successfully.")


    # 使用示例
    add_protein_id_to_json_optimized("E:/binary_ESM/dataset/dataset.json", "E:/binary_ESM/dataset/protgps_dataset_with_protein_id.json")