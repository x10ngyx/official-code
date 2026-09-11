"""Build 500-cell horizontal heatmaps after proving branch equality."""
import json
import sqlite3
import run_4gpu as suite


def main():
    out=suite.ROOT/'analysis/trace_psnr_audit'
    assert suite.read(out/'AUDIT.json')['status']=='pass'
    with sqlite3.connect(out/'traces.sqlite') as db:
        bad=db.execute("SELECT COUNT(*) FROM trace_steps a JOIN trace_steps b ON a.target=b.target AND a.sample_id=b.sample_id AND a.method=b.method AND a.step=b.step WHERE a.branch='cond' AND b.branch='uncond' AND a.reuse!=b.reuse").fetchone()[0]
        assert bad==0
        for target in suite.common.TARGETS:
            sql=f"""SELECT target,sample_id,method,'cond = uncond (verified)' AS branch,
substr(sample_id,-3)||' | '||method AS trace,step,step_label,reuse,action,
executed_blocks,skip_count,psnr,reason,threshold,skip_budget,source_trace
FROM trace_steps WHERE target={target} AND branch='cond'
ORDER BY sample_id,method,step"""
            cursor=db.execute(sql);keys=[d[0] for d in cursor.description]
            rows=[dict(zip(keys,r)) for r in cursor]
            assert len(rows)==500 and len({r['step'] for r in rows})==50 and len({r['trace'] for r in rows})==10
            payload=dict(title=f'SeaCache / SEA7 — {target:.1f}× 档，50 步逐条 trace',
                subtitle='横向 step 00–49；纵向为5条prompt×2方法。0=重算，1=复用；cond/uncond已核验相同。',
                source=dict(label='SeaCache / SEA7 measured branch traces',path=str(out/'traces.sqlite'),
                    query=dict(engine='SQLite',language='sql',sql=sql,tables_used=['trace_steps'],
                        description='Only collapse CFG branches after exact action equality; preserve every step.')),
                table=dict(rows=rows,row_count=500,truncated=False),
                chart=dict(type='heatmap',fields=dict(x={'field':'trace'},y={'field':'reuse','aggregate':'max'},color={'field':'step_label'})),
                display=dict(controls=True,unit='',x_axis_title='Prompt / method',y_axis_title='Reuse (0 or 1)'))
            suite.dump(out/f'widget_horizontal_{target:.1f}.json',payload)
    print(json.dumps(dict(status='pass',charts=3,columns=50,rows_per_chart=10,cells_per_chart=500,branch_collapse_lossless=True)))


if __name__=='__main__':main()
