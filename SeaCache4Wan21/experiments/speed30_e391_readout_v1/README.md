# SeaCache 3.0× 与 e391 K35 结果读出

只读验收既有50prompt实验，不启动推理。`audit30.py`检查SeaCache完整产物；
`compare.py`检查配对与e391产物、复算均值及逐prompt差值，生成配对CSV及审计JSON；`build_report.py`生成canonical报告输入，
使用data-analytics的deliver_portable_artifact.mjs打包自包含HTML。
审计需用既有Wan2.2环境Python执行，仅导入torch，不进行GPU推理。
结果在原SeaCache 3.0×归档的`analysis/e391_comparison/`，由项目experiment_results链接。
