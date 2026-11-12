from pathlib import Path
import os,json,cv2 as cv,numpy as np, subprocess as sp, tempfile
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter,sosfilt
cv.setNumThreads(3)
base=Path(__file__).resolve().parents[1]/'视频素材'
out=base/'完整素材_本地剪辑'
out.mkdir(exist_ok=True)
dst=out/'完整素材_人物替换_动态配乐版.mp4'
w=Path(__file__).resolve().parents[1]/'.build'
w.mkdir(exist_ok=True)
rows=json.loads((Path(__file__).with_name('镜头数据.json')).read_text('utf8'))
assets=base/'完整素材_场景与人物替换'
src=base/'完整素材.mp4'
cap=cv.VideoCapture(str(src))
W,H=1344,672
# Compare active image area, leaving the captions out of cut detection.
small=[]
while True:
 ok,fr=cap.read()
 if not ok:break
 small.append(cv.resize(fr[70:465,130:1150],(128,56),interpolation=cv.INTER_AREA))
cap.release()
if len(small)!=1373:raise ValueError('请将 1373 帧的完整素材.mp4 放入视频素材目录。')
N=len(small); fps=30;dur=N/fps
def dist(a,b):return np.abs(a.astype('float32')-b).mean()
change=np.array([0]+[dist(x,y) for x,y in zip(small,small[1:])])
starts=[0]
for a,b in zip(rows,rows[1:]):
 l,h=a['frame'],b['frame']; k=np.arange(l+1,h+1)
 mid=(l+h+1)/2
 gain=change[k]-np.minimum(19,abs(k-mid)*1.15)
 starts.append(int(k[np.argmax(gain)]) if gain.max()>14 else int(mid))
starts.append(N)
# Locally composed instrumental at 126 bpm, with distinct sections.
sr=48000; L=round(dur*sr); snd=np.zeros((L,2),np.float32);rng=np.random.default_rng(317)
def add(t,s,pan=0,vol=1):
 i=round(t*sr)
 if i>=L:return
 s=s[:min(len(s),L-i)]*vol
 snd[i:i+len(s),0]+=s*(1-pan)*.75
 snd[i:i+len(s),1]+=s*(1+pan)*.75
def tvec(d):return np.arange(round(d*sr))/sr
def osc(f,t):return np.sin(2*np.pi*f*t)
def freq(n):return 440*2**((n-69)/12)
beat=60/126
for j in range(int(dur/beat)+1):
 pos=j*beat;seq=[38,34,41,36];root=seq[(j//8)%4]; f=freq(root)
 force=.44 if pos<11.5 else .85 if pos<30 else 1.0
 if j%8==0:
  t=tvec(beat*8+.36);env=np.minimum(1,t/.36)*np.clip((t[-1]-t)/.55,0,1)
  for n,pan in [(root+12,-.35),(root+19,.35),(root+24,.08)]:
   hz=freq(n);add(pos,env*(osc(hz,t)+.43*osc(hz*1.004,t)+.22*osc(hz*2,t)),pan,.035)
 if j%2==0 or (pos>12 and j%8==7):
  t=tvec(.34);add(pos,np.sin(2*np.pi*(42*t+2.2*(1-np.exp(-34*t))))*np.exp(-t*13),0,.42*force)
 if j%4 in (1,3) and pos>7:
  t=tvec(.18);noise=sosfilt(butter(2,[1000,7400],btype='band',fs=sr,output='sos'),rng.standard_normal(len(t)))
  add(pos,(noise*.33+osc(185,t)*.22)*np.exp(-t*26),.05,.34*force)
 t=tvec(.38);add(pos, (osc(f,t)+.22*osc(f*2,t))*np.minimum(1,t/.008)*np.exp(-8*t),0,.19*force)
 for k in range(2):
  t=tvec(.09);noise=sosfilt(butter(2,6300,btype='high',fs=sr,output='sos'),rng.standard_normal(len(t)))
  add(pos+beat*.5*k,noise*np.exp(-t*65),.32 if k else -.22,.065*force)
  if j%2==0 or pos>12:
   tone=root+24+[0,7,12,7,3 if root in (38,36) else 4,7,12,14][(j*2+k)%8]
   f1=freq(tone);t=tvec(.57);v=(osc(f1,t)+.27*osc(f1*2,t))*np.minimum(1,t/.006)*np.exp(-t*9)
   add(pos+beat*.5*k,v,(-1)**k*.32,.065*force)
snd*=np.minimum(1,np.arange(L)/(.4*sr))[:,None]*np.clip(np.arange(L)[::-1]/(.8*sr),0,1)[:,None]
from scipy.io.wavfile import write
write(w/'score_full.wav',sr,(snd/max(.01,np.max(np.abs(snd)))*28000).astype(np.int16))
# Optical-flow motion fields are derived only from the original small image, never composited as character pixels.
flowimgs=[cv.cvtColor(f,cv.COLOR_BGR2GRAY) for f in small]
gridx,gridy=np.meshgrid(np.arange(W,dtype='float32'),np.arange(H,dtype='float32'))
def fit(im):
 h,ww=im.shape[:2]; ratio=W/H
 if ww/h>ratio:
  n=round(h*ratio);xx=(ww-n)//2;im=im[:,xx:xx+n]
 else:
  n=round(ww/ratio);yy=(h-n)//2;im=im[yy:yy+n]
 return cv.resize(im,(W,H),interpolation=cv.INTER_AREA if im.shape[1]>W else cv.INTER_CUBIC)
dest=w/'local_full_edit_silent.mp4'
proc=sp.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','30','-i','pipe:0','-c:v','libx264','-threads','5','-preset','veryfast','-crf','16','-pix_fmt','yuv420p',str(dest)],stdin=sp.PIPE)
cap=cv.VideoCapture(str(src)); prev=None;used=0
for j,r in enumerate(rows):
 a,b=starts[j:j+2]; anchor=r['frame']; role=r['role']
 if 'cleanfile' in r: path=assets/r['cleanfile']
 else:path=next((assets/'02_人物替换成品').glob(r['prefix']+'*.png'))
 im=fit(cv.imdecode(np.fromfile(path,np.uint8),cv.IMREAD_COLOR))
 # Feathered, bounded movement follows the tracked direction without stretching faces at cuts.
 disp=[np.zeros((56,128,2),np.float32)]
 for i in range(a+1,b):
  if change[i]>25:v=np.zeros_like(disp[0])
  else:
   v=cv.calcOpticalFlowFarneback(flowimgs[i-1],flowimgs[i],None,.5,3,17,3,5,1.1,0)
   v=cv.GaussianBlur(np.clip(v,-3,3),(0,0),5)
  disp.append(disp[-1]*.89+v)
 disp=np.stack(disp);disp-=disp[anchor-a].copy()
 live=role in ('O','E','H')
 # Keep contiguous verified background footage, falling back to the clean still near an unlabelled cut.
 good={anchor}
 if live:
  for step in [-1,1]:
   n=anchor+step
   while a<=n<b and change[max(n,n-step)]<18 and dist(small[n],small[anchor])<32:
    good.add(n);n+=step
 for i in range(a,b):
  ok,frame=cap.read();assert ok
  p=(i-a+.5)/(b-a);phase=(i-anchor)/30
  if live and i in good:
   x,y,xx,yy=r['clean_crop'];srcim=fit(frame[y:yy,x:xx]);used+=1
  else:srcim=im
  speed=1 if j%4 in (0,1,3) else -1
  zoom=float(1.034+np.clip(.046*phase*speed,-.024,.048))
  dx=float(np.clip(phase*11,-13,13))*(1 if j%3 else -1);dy=float(np.clip(phase*5,-5,5))*speed
  mx=(gridx-W/2)/zoom+W/2-dx;my=(gridy-H/2)/zoom+H/2-dy
  if not live:
   fl=cv.resize(np.tanh(disp[i-a]/3),(W,H),interpolation=cv.INTER_LINEAR)
   mx-=fl[:,:,0]*9;my-=fl[:,:,1]*6
  pic=cv.remap(srcim,mx.astype('float32',copy=False),my.astype('float32',copy=False),cv.INTER_LINEAR,borderMode=cv.BORDER_REFLECT101)
  # One to three bridge frames keep fast impacts crisp and give slower reveals a gentler arrival.
  blend=min(3 if b-a>=12 else 1,b-a-1)
  if prev is not None and i-a<blend:
   weight=.20+.80*(i-a+1)/(blend+1)
   pic=cv.addWeighted(pic,weight,cv.GaussianBlur(prev,(5,5),1.2),1-weight,0)
  proc.stdin.write(pic.tobytes())
 prev=pic
proc.stdin.close();assert proc.wait()==0
cap.release()
tmp=w/'local_scored_final.mp4'
sp.run(['ffmpeg','-y','-loglevel','error','-i',str(dest),'-i',str(src),'-i',str(w/'score_full.wav'),'-filter_complex','[1:a]loudnorm=I=-20:TP=-2:LRA=8,aresample=48000,pan=stereo|c0=c0|c1=c0[a];[2:a]loudnorm=I=-23.5:TP=-3:LRA=8,aresample=48000[b];[a][b]amix=inputs=2:normalize=0,alimiter=limit=0.90:level=0:latency=1[aout]','-map','0:v','-map','[aout]','-c:v','copy','-c:a','aac','-b:a','224k','-t',f'{dur:.6f}','-movflags','+faststart',str(tmp)],check=True)
import shutil
shutil.copyfile(tmp,dst)
print(dst,used,'native frames',N,'total')

