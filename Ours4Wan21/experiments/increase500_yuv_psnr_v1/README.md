# Same20 weighted YUV PSNR

run.py reuses all220 candidate videos and20 native baselines from final_comparison20. Output: suite/analysis/yuv_psnr20/.
Decode RGB8 via existing VideoMetrics; RGB normalized to[0,1]. BT.709 full-range Y'=0.2126R'+0.7152G'+0.0722B', Cb=(B'-Y')/1.8556+0.5, Cr=(R'-Y')/1.5748+0.5. No transfer linearization, clipping, quantization or chroma subsampling (4:4:4). This evaluates reconstructed RGB converted to YCbCr, not original encoded YUV420 planes.
Main metric: (6 PSNR_Y + PSNR_Cb + PSNR_Cr)/8, per-frame dB first, then equal frame/video means. Peak=1; MSE<1e-10 capped100dB as existing protocol. Also saves weighted-MSE PSNR separately; these definitions must not be conflated.
BT.709: https://www.itu.int/rec/R-REC-BT.709/en
6:1:1 dB example: https://www.itu.int/dms_pub/itu-s/opb/journal/S-JOURNAL-ICTF.VOL3-2020-1-P10-PDF-E.pdf
Files: manifest.json, per-frame CSV, per_video.csv, summary.csv, paired_deltas.csv, REPORT.md, VALIDATION.json. Original RGB PSNR is recomputed against prior results for every video. CPU process pool, baseline decoded once per prompt. No inference, SSIM/LPIPS or GPU work.
