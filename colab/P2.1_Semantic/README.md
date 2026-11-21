# P2.1_Semantic — Semantic LoRA (Llama-3.1-8B), seed 42

Protocol EXC-007 (2026-11-20): target chỉ gồm HDFS và BGL. Hadoop và OpenStack chỉ làm source. Mỗi fold có 3 source.

| Fold | Target | Source |
|---|---|---|
| FOLD-TARGET-ARCH-HDFS | HDFS | BGL, Hadoop, OpenStack |
| FOLD-TARGET-ARCH-BGL | BGL | HDFS, Hadoop, OpenStack |

## File

| File | Vai trò | SHA-256 |
|---|---|---|
| `P2.1_Semantic.ipynb` | Notebook chạy trên Colab | — |
| `seqlogad-code-s42-hdfs-bgl-v3.zip` | Code, build từ `../Ma_nguon/SeqLogAD` commit `d713713` (không rename trên Drive) | `feef719adcde698b507a9b0ffe46e97ca178371cc8c3d215dfd16f26b758edba` |
| `phase2-final.zip` | Data bundle semantic view của Phase 1 (không đổi) | `ebcfdfd6eb7b1607eb6d57e60fae88456d03bcb8ae2c6688ea398042c856224b` |

Để build lại code zip (tái lập được, ra đúng hash trên):

```powershell
python ..\Ma_nguon\SeqLogAD\scripts\package_colab_s42.py out.zip
```

## Chạy trên Colab

1. Upload 2 file zip vào `MyDrive/SeqLogAD/P2.1_Semantic/uploads/`.
2. Thêm `HF_TOKEN` vào Colab Secrets. Không dán token vào cell.
3. Mở `P2.1_Semantic.ipynb`, chọn GPU, rồi chọn **Run all**. Hai fold BGL rồi HDFS sẽ train lần lượt.
   Nếu Colab ngắt giữa chừng, chạy lại notebook để resume từ checkpoint trên Drive.
4. Đợi đến khi thấy `SAFE_TO_DISCONNECT_COLAB=YES`.

## Semantic output

`MyDrive/SeqLogAD/P2.1_Semantic/P2.1/seqlogad_outputs_P2.1_S42_FINAL.zip` (kèm file `.sha256`):

```
P2.1_S42/SUMMARY.json
P2.1_S42/checksums.sha256
P2.1_S42/targets/P2.1_BGL_S42.zip   (+ .sha256)  adapter LoRA-Semantic của fold BGL, gồm checkpoint tốt nhất
P2.1_S42/targets/P2.1_HDFS_S42.zip  (+ .sha256)  adapter LoRA-Semantic của fold HDFS
```

Cell cuối của notebook kiểm tra zip có đủ cả 2 target và không có file `REGISTERED_EXTERNALLY`.

Phần P2.2_Sequence sau này dùng thư mục Drive riêng `MyDrive/SeqLogAD/P2.2_Sequence`, nên không đè lên output này.
