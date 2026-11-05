import serial,time,json,re,sys
exp={e['tensor']:e for e in json.load(open('probe_expected.json'))}
s=serial.Serial("/dev/cu.usbmodem5B5E0265961",115200,timeout=1); s.dtr=False; s.rts=True; time.sleep(0.15); s.rts=False
t0=time.time(); got={}; done=False
while time.time()-t0<480 and not done:
    l=s.readline().decode(errors='replace').rstrip()
    m=re.match(r'PROBE (\d+) n=(\d+) sum=(-?\d+) first8=(.*)',l)
    if m: got[int(m.group(1))]=(int(m.group(2)),int(m.group(3)),[int(v) for v in m.group(4).replace(' ','').split(',') if v!=''])
    elif 'FAILED' in l or 'PROBE_DONE' in l or 'ALLOC' in l: print(l[:150]); done='PROBE_DONE' in l
print(len(got),'probes read in',round(time.time()-t0),'s')
json.dump({str(k):v for k,v in got.items()}, open("probe_got.json","w"))
order=[t for t in exp if t in got]
prev=0
for t in order:
    e=exp[t]; n,sm,f8=got[t]; d=(sm-e["sum"])/e["n"]; m=max(abs(a-b) for a,b in zip(f8,e["first8"]))
    flag = "  <-- jump" if abs(d)>0.5 and abs(prev)<=0.5 else ""
    print(f"{t:4d} {e['op']:18s} n={e['n']:7d} mean diff {d:+.3f} LSB, max diff in first 8: {m}{flag}"); prev=d
