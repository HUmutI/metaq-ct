# -*- coding: utf-8 -*-
"""Context-Routed CT pipeline figure. Layout is generated and checked:
   text fits its box · text never sits on artwork · text never overlaps text
   · edges never cross an unrelated box · every edge endpoint lands on a box edge."""
W,H = 1760, 940
PAD = 13
NODES,ART,EDGES,FREE,ERRS = [],[],[],[],[]
AW = {'hd':7.7,'lbl':7.0,'s':6.0,'mono':6.35}
def wof(t,st): return len(t)*AW[st]
def esc(t): return t.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
ALT=('Context-Routed CT pipeline. Top lane: a chest CT volume, a three-channel view, '
 'TotalSegmentator producing ten anatomy regions, and a ResNet-18 encoder producing a spatial '
 'feature map. Centre: the AnatomyQFormer query bank with anatomy, general pathology, '
 'conditioned pathology and global strips, separated by a self-attention group mask. Lower gold '
 'lane: indication text through a frozen CXR-BERT, with age-band and sex embeddings, forming a '
 'context token sequence that conditions the conditioned queries (C1) and produces pool weights '
 '(C2). Right: Z_gen and Z_ind pools concatenated into Z_final and fed to the shared embedding '
 'space. Bottom lane: radiology reports through Qwen3-8B and CXR-BERT with LoRA into the same space.')

def box(nid,x,y,w,h,kind,lines=(),rx=4,label=None,lb=None):
    for t,st in lines:
        if wof(t,st) > w-2*PAD: ERRS.append('%s: "%s" needs %.0f has %.0f'%(nid,t,wof(t,st),w-2*PAD))
    if label:
        t,st = label
        if wof(t,st) > w-2*PAD: ERRS.append('%s label "%s" too wide'%(nid,t))
    NODES.append(dict(id=nid,x=x,y=y,w=w,h=h,kind=kind,lines=list(lines),rx=rx,label=label,lb=lb))
    return dict(id=nid,x=x,y=y,w=w,h=h,cx=x+w/2,cy=y+h/2,r=x+w,b=y+h)
def art(kind,x,y,w,h,**kw): ART.append(dict(kind=kind,x=x,y=y,w=w,h=h,**kw))
def free(x,y,t,st='mono',anchor='start',cls=''): FREE.append((x,y,t,st,anchor,cls))
def edge(pts,kind='s',dash=False,head=True): EDGES.append(dict(pts=pts,kind=kind,dash=dash,head=head))

# ══════════════════ image lane ══════════════════
R1,H1 = 74,100
R2,H2 = 206,78
ct  = box('ct',  40,R1,120,H1,'img',label=('chest CT volume','s'),lb=R1+H1-12)
art('cube',  56,R1+14, 88,54)
ch3 = box('ch3',192,R1,120,H1,'img',label=('3-channel view','s'),lb=R1+H1-12)
art('planes',206,R1+14, 92,54)
rn  = box('rn', 344,R1,156,H1,'old',label=('ResNet-18 3D','lbl'),lb=R1+H1-12)
art('net',  372,R1+12,100,56)
fm  = box('fm', 532,R1,140,H1,'img',label=('F · 12³ × 512','mono'),lb=R1+H1-12)
art('grid', 548,R1+14,108,52,cols=6,rows=4)
ts  = box('ts', 344,R2,156,H2,'old',label=('TotalSegmentator','lbl'),lb=R2+H2-11)
art('seg',  358,R2+11,128,32)
mk  = box('mk', 532,R2,140,H2,'old',label=('role mask m · ρ','mono'),lb=R2+H2-11)
art('patch',546,R2+11,112,32)

edge([(ct['r'],ct['cy']),(ch3['x'],ct['cy'])])
edge([(ch3['r'],ch3['cy']),(rn['x'],ch3['cy'])])
edge([(ch3['cx'],ch3['b']),(ch3['cx'],ts['cy']),(ts['x'],ts['cy'])])
edge([(rn['r'],rn['cy']),(fm['x'],rn['cy'])])
edge([(ts['r'],ts['cy']),(mk['x'],ts['cy'])])

# ══════════════════ Q-Former ══════════════════
QFX,QFW,QFY,QFH = 712,376,62,344
qf = box('qf',QFX,QFY,QFW,QFH,'qf')
free(QFX+PAD, QFY+26,'AnatomyQFormer','hd')
free(QFX+PAD, QFY+44,'67 slots · 40 distinct queries','mono')
SX,SW,SH = QFX+PAD,QFW-2*PAD,27
def strip(i,label,cnt,kind,bar):
    y = QFY+58 + i*(SH+6)
    b = box('st%d'%i,SX,y,SW,SH,kind)
    free(SX+11,y+18,label,'s','start','g' if kind=='new' else '')
    free(SX+SW-11,y+18,cnt,'mono','end','g' if kind=='new' else '')
    if bar: art('bar',SX+SW-11-wof(cnt,'mono')-16-88,y+8,88,11,gold=(kind=='new'))
    return b
s0=strip(0,'anatomy','× 10','old',True)
s1=strip(1,'pathology · general','× 27','old',True)
s2=strip(2,'pathology · conditioned','× 27','new',True)
y3 = QFY+58+3*(SH+6)
box('st3',SX,y3,180,SH,'old');       free(SX+11,y3+18,'global','s');   free(SX+169,y3+18,'× 2','mono','end')
box('st4',SX+190,y3,SW-190,SH,'new');free(SX+201,y3+18,'clinical','s','start','g'); free(SX+SW-11,y3+18,'× 1','mono','end','g')
edge([(SX,y3+SH+16),(SX+240,y3+SH+16)],'g',dash=True,head=False)
free(SX,y3+SH+34,'group mask · general ↛ conditioned','mono','start','g')
ca = box('ca',SX,y3+SH+44,240,30,'old',label=('cross-attention → F','s'))
free(SX,ca['b']+20,'ResNet is never conditioned','mono')
free(SX,ca['b']+36,'pooling is group-restricted','mono')
edge([(fm['r'],fm['cy']),(QFX,fm['cy'])])
edge([(mk['r'],mk['cy']),(mk['r']+20,mk['cy']),(mk['r']+20,ca['cy']),(QFX,ca['cy'])])

# ══════════════════ pools ══════════════════
PX,PW = 1200,190
zg = box('zg',PX,110,PW,58,'old',[('Z_gen','hd'),('39 tokens · no I','mono')])
zi = box('zi',PX,196,PW,70,'new',[('Z_ind','hd'),('Σ w·z / Σ w','mono'),('w = 1 + βr   (β ≥ 0)','mono')])
zf = box('zf',PX,300,PW,54,'old',[('Z_final = W[Z_g;Z_i]','mono'),('W init [ I ; 0 ]','mono')])
edge([(qf['x']+QFW,zg['cy']),(PX,zg['cy'])])
edge([(qf['x']+QFW,zi['cy']),(PX,zi['cy'])],'g')
edge([(zg['x']+36,zg['b']),(zg['x']+36,180),(1160,180),(1160,zf['cy']-9),(zf['x'],zf['cy']-9)])
edge([(PX+142,zi['b']),(PX+142,zf['y'])],'g')

# ══════════════════ shared space ══════════════════
SHX,SHW = 1440,208
sh = box('sh',SHX,94,SHW,206,'qf')
free(SHX+PAD,118,'shared space','hd')
art('mat',SHX+PAD,128,96,96)
for i,(t,c) in enumerate([('L_clip',''),('L_cls',''),('L_org',''),('L_ptok',''),('L_ind','g'),('L_cf','g')]):
    free(SHX+124,142+i*17,t,'mono','start',c)
free(SHX+PAD,246,'zero-shot · 27 prompts','mono')
free(SHX+PAD,262,'mask-free inference','mono')
edge([(zg['r'],zg['cy']),(SHX,zg['cy'])])
edge([(zf['r'],zf['cy']),(1404,zf['cy']),(1404,238),(SHX,238)])

# ══════════════════ context lane (new) ══════════════════
LY = 490
free(40,LY-22,'NEW · CONTEXT PATHWAY','mono','start','secg')
ind = box('ind', 40,LY,196,56,'new',[('indication','lbl'),('free text · ~21 words','mono')])
age = box('age', 40,LY+70, 94,48,'new',[('age → band','s'),('10 bands','mono')])
sex = box('sex',142,LY+70, 94,48,'new',[('sex','s'),('+ interact','mono')])
bert= box('bert',268,LY,158,56,'new',label=('CXR-BERT · frozen','s'),lb=LY+50)
art('net',300,LY+5,94,28)
hc  = box('hc', 458,LY,174,56,'new',[('H_C = [H_ind;H_dem]','mono'),('token sequence','mono')])
c1  = box('c1', 664,LY-16,220,64,'new',[('C1 · condition','hd'),('q̃ = q + CrossAttn(q,H_C)','mono'),('+ FiLM · zero-init','mono')])
c2  = box('c2', 664,LY+72,220,52,'new',[('C2 · relevance','hd'),('r = σ MLP[E(I),E(P_c)]','mono')])
edge([(ind['r'],ind['cy']),(bert['x'],ind['cy'])],'g')
edge([(sex['r'],sex['cy']),(sex['r']+16,sex['cy']),(sex['r']+16,ind['cy']+16),(bert['x'],ind['cy']+16)],'g')
edge([(bert['r'],bert['cy']),(hc['x'],bert['cy'])],'g')
edge([(hc['r'],hc['cy']-10),(c1['x'],c1['cy'])],'g')
edge([(hc['r'],hc['cy']+10),(c2['x']-18,hc['cy']+10),(c2['x']-18,c2['cy']),(c2['x'],c2['cy'])],'g')
CH = QFX+QFW+22
edge([(c1['r'],c1['cy']),(CH,c1['cy']),(CH,s2['cy']),(SX+SW,s2['cy'])],'g')
free(CH-10,c1['cy']-10,'C1','mono','end','g')
edge([(c2['r'],c2['cy']),(1420,c2['cy']),(1420,zi['cy']),(zi['r'],zi['cy'])],'g')
free(1412,c2['cy']-10,'C2','mono','end','g')

# ══════════════════ report lane ══════════════════
RY = 728
free(40,RY-22,'INHERITED · REPORT PATHWAY','mono','start','sec')
r0 = box('r0', 40,RY,178,56,'txt',[('radiology report','lbl'),('findings + impressions','mono')])
r1 = box('r1',252,RY,150,56,'txt',[('Qwen3-8B','lbl'),('labels + regions','mono')])
r2 = box('r2',436,RY,176,56,'txt',[('per-organ sentences','lbl'),('region_cache','mono')])
r3 = box('r3',646,RY,172,56,'txt',label=('CXR-BERT + LoRA','s'),lb=RY+50)
art('net',680,RY+5,104,28)
r4 = box('r4',852,RY,166,56,'txt',[('text embeddings','lbl'),('per-organ + global','mono')])
for a,b in [(r0,r1),(r1,r2),(r2,r3),(r3,r4)]: edge([(a['r'],a['cy']),(b['x'],a['cy'])],'b')
edge([(r4['r'],r4['cy']),(SHX+SHW-46,r4['cy']),(SHX+SHW-46,sh['b'])],'b')

# ══════════════════ footer ══════════════════
FY = 838
box('foot',40,FY,W-80,66,'ghost')
free(56,FY+22,'Three structural layers protect a finding; a fourth fixes the starting point','s')
free(56,FY+41,'1 · residual  q̃ = q + Δ     2 · w ≥ 1, no class attenuated     3 · Z_gen isolated — group mask, group pooling, λ blind to I','mono')
free(56,FY+58,'4 · at step 0 every gate is zero and W = [I;0]   ⇒   Z_final = Z_gen = ARC-CT, bit for bit','mono','start','g')

# ══════════════════ checks ══════════════════
def tb(x,y,t,st,anc): 
    w=wof(t,st); return (x-w if anc=='end' else x, y-11, w, 14, t)
FB=[tb(x,y,t,st,a) for x,y,t,st,a,c in FREE]
for n in NODES:
    if n['label']:
        t,st=n['label']; _b=n['lb'] if n['lb'] else n['y']+n['h']/2+4
        FB.append((n['x']+PAD,_b-11,wof(t,st),14,t))
    yy=n['y']+22 if len(n['lines'])>1 else n['y']+n['h']/2+4
    for t,st in n['lines']:
        FB.append((n['x']+PAD,yy-11,wof(t,st),14,t)); yy+=17
for i,(ax,ay,aw,ah,at) in enumerate(FB):
    if ax<0 or ax+aw>W: ERRS.append('text off-canvas: "%s"'%at[:30])
    for bx,by,bw,bh,bt in FB[i+1:]:
        if ax<bx+bw and bx<ax+aw and ay<by+bh and by<ay+ah: ERRS.append('text∩text: "%s" / "%s"'%(at[:22],bt[:22]))
    for g in ART:
        if ax<g['x']+g['w'] and g['x']<ax+aw and ay<g['y']+g['h'] and g['y']<ay+ah:
            ERRS.append('text∩art(%s): "%s"'%(g['kind'],at[:24]))
for n in NODES:
    if n['x']+n['w']>W or n['y']+n['h']>H: ERRS.append('node off-canvas %s'%n['id'])
for g in ART:
    if g['x']+g['w']>W or g['y']+g['h']>H: ERRS.append('art off-canvas %s'%g['kind'])
rects=[(n['x'],n['y'],n['w'],n['h'],n['id']) for n in NODES if n['kind']!='ghost']
def on_edge(p):
    x,y=p
    for bx,by,bw,bh,bid in rects:
        if (abs(x-bx)<2 or abs(x-(bx+bw))<2) and by-2<=y<=by+bh+2: return True
        if (abs(y-by)<2 or abs(y-(by+bh))<2) and bx-2<=x<=bx+bw+2: return True
    return False
for e in EDGES:
    if not e['head']: continue
    if not on_edge(e['pts'][0]):  ERRS.append('edge start floating at %s'%(e['pts'][0],))
    if not on_edge(e['pts'][-1]): ERRS.append('edge end floating at %s'%(e['pts'][-1],))
    for (x1,y1),(x2,y2) in zip(e['pts'],e['pts'][1:]):
        for bx,by,bw,bh,bid in rects:
            if bid in ('qf','sh'): continue
            if x1==x2 and bx+3<x1<bx+bw-3 and not(max(y1,y2)<=by+3 or min(y1,y2)>=by+bh-3): ERRS.append('edge∩%s v'%bid)
            if y1==y2 and by+3<y1<by+bh-3 and not(max(x1,x2)<=bx+3 or min(x1,x2)>=bx+bw-3): ERRS.append('edge∩%s h'%bid)
if ERRS:
    print('LAYOUT ERRORS (%d):'%len(set(ERRS)))
    for e in sorted(set(ERRS)): print('  ',e)
    raise SystemExit(1)
print('layout OK · %d boxes · %d artworks · %d edges'%(len(NODES),len(ART),len(EDGES)))

# ══════════════════ render ══════════════════
FILL={'old':('#151D26','#3E4E60',1.1),'new':('#241E12','#D9A441',1.3),
      'qf':('#171630','#7E6FC4',1.2),'txt':('#1D1729','#B07CC6',1.1),
      'img':('#12202F','#5B8FD4',1.1),'ghost':('none','#26313D',1.0)}
STROKE={'s':'#6B7A8C','g':'#D9A441','b':'#7E6FC4'}
o=[]

def draw_art(a):
    k,x,y,w,h=a['kind'],a['x'],a['y'],a['w'],a['h']
    r=[]
    if k=='cube':
        d=w*.20
        r.append('<path d="M%g %g h%g l%g %g v%g l%g %g h%g z" fill="#1B2A3C" stroke="#5B8FD4" stroke-width="1"/>'
                 %(x,y+d, w-d, d,-d, h-d, -d,d, -(w-d)))
        r.append('<path d="M%g %g l%g %g v%g l%g %g z" fill="#24374C" stroke="#5B8FD4" stroke-width=".8"/>'%(x+w-d,y+d,d,-d,h-d,-d,d))
        r.append('<ellipse cx="%g" cy="%g" rx="%g" ry="%g" fill="#3E5C7E" opacity=".55"/>'%(x+(w-d)/2,y+d+(h-d)/2,(w-d)*.30,(h-d)*.26))
        r.append('<ellipse cx="%g" cy="%g" rx="%g" ry="%g" fill="#16283A"/>'%(x+(w-d)/2,y+d+(h-d)/2,(w-d)*.13,(h-d)*.15))
    elif k=='planes':
        for i,(dx,dy,c) in enumerate([(14,0,'#7E3B44'),(7,13,'#3B7E4E'),(0,26,'#3B5B8E')]):
            r.append('<rect x="%g" y="%g" width="%g" height="%g" rx="2.5" fill="#16232F" stroke="%s" stroke-width="1.2"/>'%(x+dx,y+dy,w-16,20,c))
            r.append('<ellipse cx="%g" cy="%g" rx="%g" ry="6" fill="#2C4560" opacity=".8"/>'%(x+dx+(w-16)/2,y+dy+10,(w-16)*.26))
    elif k=='net':
        r.append('<path d="M%g %g l%g %g v%g l%g %g z" fill="#16202C" stroke="#5B8FD4" stroke-width="1"/>'%(x,y,w*.22,h*.18,h*.64,-w*.22,h*.18))
        r.append('<path d="M%g %g l%g %g v%g l%g %g z" fill="#16202C" stroke="#5B8FD4" stroke-width="1"/>'%(x+w,y,-w*.22,h*.18,h*.64,w*.22,h*.18))
        for i in range(3):
            for j in range(3):
                r.append('<line x1="%g" y1="%g" x2="%g" y2="%g" stroke="#33506E" stroke-width=".6"/>'
                         %(x+w*.24,y+h*.28+i*h*.22, x+w*.76,y+h*.28+j*h*.22))
        for i in range(3):
            r.append('<circle cx="%g" cy="%g" r="3.1" fill="#0F1A26" stroke="#6FA0D6" stroke-width="1"/>'%(x+w*.24,y+h*.28+i*h*.22))
            r.append('<circle cx="%g" cy="%g" r="3.1" fill="#0F1A26" stroke="#6FA0D6" stroke-width="1"/>'%(x+w*.76,y+h*.28+i*h*.22))
    elif k=='grid':
        cw,ch_=w/a['cols'],h/a['rows']
        hot={(1,1),(3,2),(4,0)}
        for i in range(a['rows']):
            for j in range(a['cols']):
                f='#4E7CB0' if (j,i) in hot else '#26405C'
                r.append('<rect x="%g" y="%g" width="%g" height="%g" fill="%s" stroke="#5B8FD4" stroke-width=".5"/>'%(x+j*cw,y+i*ch_,cw,ch_,f))
    elif k=='seg':
        cols=['#4E7A56','#7A5A44','#4A5E86','#6B4E76','#7A7248','#48706E','#7A4E56','#566E48']
        cw=w/8
        for j,c in enumerate(cols):
            hh=h*(.55+.11*((j*5)%4))
            r.append('<rect x="%g" y="%g" width="%g" height="%g" rx="2" fill="%s" opacity=".9"/>'%(x+j*cw+1.5,y+(h-hh)/2,cw-3,hh,c))
    elif k=='patch':
        cols=['#3D5C46','#5C4636','#3A4A63','#54465F','#5A5236','#3E5757']
        cw=w/3
        for j in range(3):
            for i in range(2):
                r.append('<rect x="%g" y="%g" width="%g" height="%g" rx="2" fill="%s"/>'%(x+j*cw+1.5,y+i*(h/2)+1.5,cw-3,h/2-3,cols[j+i*3]))
    elif k=='bar':
        r.append('<rect x="%g" y="%g" width="%g" height="%g" rx="2" fill="%s"/>'%(x,y,w,h,'#D9A441' if a.get('gold') else '#4E5D6E'))
    elif k=='mat':
        for i in range(4):
            for j in range(4):
                f='#5A4FA8' if i==j else '#2C2850'
                r.append('<rect x="%g" y="%g" width="22" height="22" fill="%s" stroke="#463E78" stroke-width=".5"/>'%(x+j*24,y+i*24,f))
    return r

o.append('<svg viewBox="0 0 %d %d" role="img" aria-label="%s">'%(W,H,ALT))
o.append('<defs>')
for k,col in STROKE.items():
    o.append('<marker id="a%s" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse"><path d="M0,1 L10,5 L0,9 Z" fill="%s"/></marker>'%(k,col))
o.append('</defs>')
o.append('<style>.hd{font-family:"Public Sans",sans-serif;font-size:13.5px;font-weight:600;fill:#F2F6FA}'
 '.lbl{font-family:"Public Sans",sans-serif;font-size:13px;fill:#DDE4EB}'
 '.s{font-family:"Public Sans",sans-serif;font-size:11px;fill:#9AA8B6}'
 '.mono{font-family:"JetBrains Mono",monospace;font-size:10.5px;fill:#6F7E8F}'
 '.mono.g,.s.g{fill:#D9A441}.mono.sec{fill:#55626D;letter-spacing:.18em}'
 '.mono.secg{fill:#8A6B2E;letter-spacing:.18em}</style>')
o.append('<text class="mono sec" x="40" y="42">INHERITED · ARC-CT</text>')
o.append('<line x1="40" y1="54" x2="%d" y2="54" stroke="#212B36"/>'%(W-40))
o.append('<line x1="40" y1="%d" x2="%d" y2="%d" stroke="#3A2F1C"/>'%(LY-14,W-40,LY-14))
o.append('<line x1="40" y1="%d" x2="%d" y2="%d" stroke="#212B36"/>'%(RY-14,W-40,RY-14))
for n in NODES:
    f,st,sw=FILL[n['kind']]
    dash=' stroke-dasharray="4 4"' if n['kind']=='ghost' else ''
    o.append('<rect x="%g" y="%g" width="%g" height="%g" rx="%g" fill="%s" stroke="%s" stroke-width="%g"%s/>'
             %(n['x'],n['y'],n['w'],n['h'],n['rx'],f,st,sw,dash))
for a in ART: o += draw_art(a)
for n in NODES:
    g=' g' if n['kind']=='new' else ''
    if n['label']:
        t,st=n['label']; b=n['lb'] if n['lb'] else n['y']+n['h']/2+4
        o.append('<text class="%s%s" x="%g" y="%g">%s</text>'%(st,g if st=='s' else '',n['x']+PAD,b,esc(t)))
    yy=n['y']+22 if len(n['lines'])>1 else n['y']+n['h']/2+4
    for t,st in n['lines']:
        o.append('<text class="%s%s" x="%g" y="%g">%s</text>'%(st,g if st in('s','mono') else '',n['x']+PAD,yy,esc(t))); yy+=17
for x,y,t,st,anc,cls in FREE:
    a=' text-anchor="end"' if anc=='end' else ''
    o.append('<text class="%s%s" x="%g" y="%g"%s>%s</text>'%(st,(' '+cls) if cls else '',x,y,a,esc(t)))
for e in EDGES:
    d='M'+' L'.join('%g %g'%p for p in e['pts'])
    o.append('<path d="%s" fill="none" stroke="%s" stroke-width="%s"%s%s/>'
             %(d,STROKE[e['kind']],'1.7' if e['kind']=='g' else '1.4',
               ' stroke-dasharray="4 3.5"' if e['dash'] else '',
               ' marker-end="url(#a%s)"'%e['kind'] if e['head'] else ''))
o.append('</svg>')
open('_fig.svg','w',encoding='utf-8').write('\n'.join(o))
print('svg: %d byte'%len('\n'.join(o)))
