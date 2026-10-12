# -*- coding: utf-8 -*-
"""
把本地文件上传到百度 AI Studio 数据集。

流程（实测有效的顺序）：
  1) GET  /llm/files/acl               -> 一次性 BOS 临时凭证 + 目标 fileKey
  2) BOS  upload (multipart)           -> 上传文件体到 bucket
  3) POST /llm/files/addfile           -> 落库，拿 fileId
  4) POST /llm/files/datasets/{id}/addfile -> 挂进指定数据集

用法：
  python tools/aistudio_upload.py --file <本地文件> --dataset-id 400144 [--apply]

不带 --apply 只干跑。
"""
import argparse
import sys
from pathlib import Path

import requests

API = "https://aistudio.baidu.com"
TOKEN_FILE = Path.home() / ".aistudio_token"
MB = 1024 * 1024


def load_token(explicit=None):
    if explicit:
        return explicit.strip()
    import os
    env = os.getenv("AISTUDIO_ACCESS_TOKEN")
    if env:
        return env.strip()
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text(encoding="utf-8").strip()
    return None


def H(token):
    return {"Authorization": token, "Content-Type": "application/json"}


def get_acl(token, timeout=30):
    """返回 result: accessKeyId/secretAccessKey/sessionToken/fileKey/endpoint/bucketName"""
    r = requests.get(f"{API}/llm/files/acl", headers=H(token), timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if j.get("errorCode") != 0:
        raise RuntimeError(f"acl 失败: {j.get('errorMsg')}")
    return j["result"]


def register_file(token, origin_name, file_key, file_abs, bucket_name, timeout=60):
    data = {
        "fileOriginName": origin_name,
        "fileKey": file_key,
        "bucketName": bucket_name,
        "fileAbs": file_abs,
    }
    r = requests.post(f"{API}/llm/files/addfile", headers=H(token), json=data, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if j.get("errorCode") != 0:
        raise RuntimeError(f"落库失败: {j.get('errorMsg')}  {r.text[:300]}")
    return j["result"]["fileId"]


def create_dataset_with_file(token, dataset_name, file_ids, dataset_abs="", dataset_type=2,
                             protocol_id=1, timeout=60):
    """建新数据集并直接带上文件（实测 addfile 单独接口会 500，建库时带 fileIds 有效）。"""
    data = {
        "datasetName": dataset_name,
        "datasetAbs": dataset_abs,
        "fileIds": file_ids,
        "datasetType": dataset_type,
        "protocolId": protocol_id,
    }
    r = requests.post(f"{API}/llm/files/datasets", headers=H(token), json=data, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if j.get("errorCode") != 0:
        raise RuntimeError(f"建库失败: {j.get('errorMsg')}  {r.text[:300]}")
    return j["result"]["datasetId"]


def bos_client(acl):
    from baidubce.auth.bce_credentials import BceCredentials
    from baidubce.bce_client_configuration import BceClientConfiguration
    from baidubce.services.bos.bos_client import BosClient

    # 临时凭证必须通过 security_token 传入，否则签名校验失败
    cfg = BceClientConfiguration(
        credentials=BceCredentials(acl["accessKeyId"], acl["secretAccessKey"]),
        endpoint=acl["endpoint"],
        security_token=acl["sessionToken"],
    )
    return BosClient(cfg)


def bos_upload(acl, local_path, file_key, part_mb=16):
    """分片上传。返回 None（BOS 端幂等：同名 key 覆盖）。"""
    client = bos_client(acl)
    bucket = acl["bucketName"]
    size = local_path.stat().st_size
    part = part_mb * MB

    if size <= part:
        client.put_object_from_file(bucket, file_key, str(local_path))
        print(f"  单片上传完成 ({size/MB:.1f} MB)")
        return

    n_parts = (size + part - 1) // part
    up = client.initiate_multipart_upload(bucket, file_key)
    upload_id = up.upload_id
    parts = []
    try:
        for i in range(1, n_parts + 1):
            offset = (i - 1) * part
            length = min(part, size - offset)
            resp = client.upload_part_from_file(
                bucket, file_key, upload_id, i, length, str(local_path), offset,
            )
            # eTag 在响应头里：resp.to_map()['headers']['etag']
            etag = (resp.to_map().get("headers") or {}).get("etag")
            if not etag:
                raise RuntimeError(f"分片 {i} 未返回 eTag: {resp.to_map()}")
            parts.append({"partNumber": i, "eTag": etag})
            done = offset + length
            print(f"\r  分片 {i:>3}/{n_parts}  {done/MB:8.1f}/{size/MB:.1f} MB "
                  f"({done*100.0/size:5.1f}%)", end="", flush=True)
        print()
        client.complete_multipart_upload(bucket, file_key, upload_id, parts)
    except Exception:
        try:
            client.abort_multipart_upload(bucket, file_key, upload_id)
        except Exception:
            pass
        raise


def main():
    ap = argparse.ArgumentParser(description="上传文件到 AI Studio 数据集")
    ap.add_argument("--file", required=True)
    ap.add_argument("--dataset-id", type=int, default=None,
                    help="已有数据集 ID；给了就直接挂进去，不给则新建")
    ap.add_argument("--new-dataset", default=None,
                    help="新建数据集的名称（如 lab-ppe-kaggle）；与 --dataset-id 二选一")
    ap.add_argument("--name", default=None, help="远端文件名（默认同本地名）")
    ap.add_argument("--part-mb", type=int, default=16)
    ap.add_argument("--token", default=None)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    token = load_token(args.token)
    if not token:
        print("[错误] 无 token")
        return 2
    src = Path(args.file).resolve()
    if not src.is_file():
        print(f"[错误] 文件不存在: {src}")
        return 2
    name = args.name or src.name
    if not args.dataset_id and not args.new_dataset:
        print("[错误] 需要 --dataset-id 或 --new-dataset 之一")
        return 2

    print(f"文件     : {src}")
    print(f"大小     : {src.stat().st_size/MB:.1f} MB")
    print(f"远端名   : {name}")
    print(f"目标     : " + (f"挂到已有数据集 {args.dataset_id}" if args.dataset_id
                           else f"新建数据集 {args.new_dataset}"))

    print("\n[1/4] 申请 BOS 凭证 + fileKey ...")
    acl = get_acl(token)
    print(f"      bucket   = {acl['bucketName']}")
    print(f"      endpoint = {acl['endpoint']}")
    print(f"      fileKey  = {acl['fileKey']}")

    if not args.apply:
        print("\n[干跑] 加 --apply 真正执行。")
        return 0

    print(f"\n[2/4] 上传到 BOS（分片 {args.part_mb}MB）...")
    bos_upload(acl, src, acl["fileKey"], args.part_mb)

    print("\n[3/4] 落库登记 ...")
    fid = register_file(token, name, acl["fileKey"], str(src), acl["bucketName"])
    print(f"      fileId = {fid}")

    if args.dataset_id:
        print(f"\n[4/4] 挂到数据集 {args.dataset_id} ...")
        attach_to_dataset(token, args.dataset_id, [fid])
        print("      完成")
    else:
        print(f"\n[4/4] 新建数据集「{args.new_dataset}」并带入文件 ...")
        did = create_dataset_with_file(
            token, args.new_dataset, [fid],
            dataset_abs="Lab-PPE YOLOv8 4 类检测数据集（mask/gloves/lab_coat/goggles）",
        )
        print(f"      datasetId = {did}")
        print(f"      链接：https://aistudio.baidu.com/dataset/detail/{did}/intro")

    print("\n[成功] 上传结束。回网页刷新数据集详情页即可看到文件。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
