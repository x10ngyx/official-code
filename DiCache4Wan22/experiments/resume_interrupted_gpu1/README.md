# Resume interrupted GPU1 without restarting GPU2/3

recover.py validates frozen original sources and completed GPU1 manifests, starts original --worker 1 --resume, and adopts GPU2/3 using PID/start-time identity and completion markers. The stale original coordinator remains stopped until all generation workers exit successfully, then is retired without its failure cleanup. Recovery then acquires the original lock and executes the original three quality finalizers and root completion checks. No change to frozen generation code, assignments, model settings or valid results.

--self-test checks eight completion/identity guards. Use wan2.2 Python, four numerical thread variables=1, --output-dir <existing external result> and --old-coordinator <paused PID>. Output/heartbeat/precheck logs are under existing result gpu1_recovery/. A failed worker is reported without terminating healthy workers. Intended for this explicit one-worker recovery.
