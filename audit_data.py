"""Inspect every recovered GOOG/NSE input without conflating quote updates with trades."""
from pathlib import Path
import json,hashlib
import numpy as np
root=Path(__file__).resolve().parent
rows=[]
for p in sorted((root/'data').rglob('*.npz')):
 with np.load(p,allow_pickle=False) as z:
  ts=z['ts'];row=dict(path=str(p.relative_to(root)),sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest(),rows=len(ts),first_timestamp_ms=int(ts.min()),last_timestamp_ms=int(ts.max()),backward_timestamps=int(np.sum(np.diff(ts)<0)),duplicate_timestamps=int(np.sum(np.diff(ts)==0)))
  if 'bb' in z:
   bid,ask=z['bb'],z['ba'];row.update(clock='book_updates',crossed_or_locked=int(np.sum(bid>=ask)),nonfinite_price=int(np.sum(~np.isfinite(bid)|~np.isfinite(ask))),negative_depth=int(np.sum((z['bq']<0)|(z['aq']<0))))
  else:
   row.update(clock='snapshot_linked_trades',trade_prints=len(z['tr_bookidx']),crossed_or_locked=int(np.sum(z['bids'][:,0,0]>=z['asks'][:,0,0])),invalid_trade_mapping=int(np.sum((z['tr_bookidx']<0)|(z['tr_bookidx']>=len(ts)))))
  rows.append(row)
(root/'verification/data_audit.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
