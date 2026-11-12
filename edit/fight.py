from pathlib import Path
import json,os,cv2 as cv,numpy as np
from PIL import Image,ImageDraw
w=Path(__file__).resolve().parents[1]/'.build'
w.mkdir(exist_ok=True)
base=Path(__file__).resolve().parents[1]/'视频素材';original=base/'完整素材.mp4';folder=base/'完整素材_本地剪辑';old=folder/'完整素材_人物替换_动态配乐版.mp4'
rows=json.loads((Path(__file__).with_name('镜头数据.json')).read_text('utf8'))
by={r['num']:r for r in rows}
cv.setNumThreads(3)
from scipy.interpolate import RBFInterpolator,PchipInterpolator
import subprocess as sp

W,H=1344,672
# Coordinates in the 250 x 125 shot board. Each control specifies a radius and motion from windup to contact.
clips=[]
def shot(n,a,b,cam,controls,impact=.67,mode='hit'):
 clips.append((n,a,b,cam,controls,impact,mode))
# radius 0 is a joint pin; radii support rigid islands around hands, heads and chests.
shot(29,382,394,(3,-1),[(57,33,13,11,7),(86,20,10,3,2),(120,22,5,-5,4),(169,32,13,-3,2),(98,95,19,3,-1)],.66)
shot(30,394,405,(5,-3),[(117,88,10,15,-10),(155,57,12,19,-14),(167,30,11,6,-3),(202,40,6,8,-5),(170,101,14,2,-1)],.64)
shot(32,405,419,(-5,1),[(47,78,17,-19,3),(116,70,10,-13,3),(166,57,5,-5,2),(208,46,12,-4,1),(218,77,6,-4,2),(204,110,11,0,0)],.8)
shot(34,419,431,(-4,1),[(103,67,9,-10,-2),(168,62,12,-5,0),(216,45,12,-3,0),(78,61,7,-8,1),(66,42,12,-5,2),(33,91,11,-3,1)],.37)
shot(35,431,441,(3,-1),[(129,38,8,9,-6),(100,20,9,4,-2),(90,73,11,1,-1),(170,98,11,2,1),(77,105,14,0,0)],.59)
shot(55,754,767,(4,2),[(52,51,8,1,-2),(105,50,10,7,5),(163,96,15,13,10),(53,108,15,-1,0)],.68)
shot(57,767,787,(5,-2),[(179,46,7,17,-6),(124,55,8,10,-4),(143,22,12,4,-2),(184,83,13,8,4),(204,110,7,6,4),(59,100,11,-2,1)],.65)
shot(59,792,802,(-1,2),[(137,42,9,-3,5),(120,72,8,-4,4),(97,77,6,-2,1),(136,90,6,0,0)],.43)
shot(67,944,956,(-2,-2),[(93,49,12,-4,2),(97,95,8,3,-5),(229,89,11,-4,-6),(156,99,12,0,0)],.75)
shot(69,964,975,(-5,1),[(30,82,12,-8,2),(62,92,10,-6,1),(136,70,14,-13,4),(196,62,10,-6,2)],.8)
shot(70,975,986,(5,-1),[(66,56,13,5,0),(124,69,8,10,-2),(186,57,14,18,-4),(35,93,7,0,0)],.61)
shot(71,986,1001,(-2,1),[(136,72,11,4,3),(107,99,14,2,1),(112,24,18,1,1)],.49)
shot(72,1001,1014,(4,1),[(105,56,11,10,3),(50,43,12,6,1),(186,76,13,7,4),(147,59,10,4,2)],.55)
shot(73,1014,1034,(3,2),[(103,49,12,4,3),(141,71,7,8,5),(157,91,6,14,8),(58,78,12,0,1)],.64)
shot(80,1088,1097,(4,-2),[(145,85,7,6,-5),(165,53,12,8,-5),(111,74,9,4,-3),(74,101,11,1,-1)],.36)
shot(81,1097,1113,(-1,1),[(108,89,6,2,3),(69,54,10,-3,-1),(161,44,11,4,-2),(190,90,8,2,2)],.4)
shot(84,1144,1165,(3,-2),[(127,25,11,6,-2),(117,43,7,10,-3),(150,70,13,17,-8),(101,86,10,6,-3),(193,45,10,5,-2),(179,53,6,8,-4),(203,72,11,4,0)],.61)
shot(86,1165,1191,(2,-2),[(115,44,10,3,-4),(92,43,6,-3,-4),(142,24,5,4,-5),(151,67,8,8,-5),(107,103,6,1,5)],.54)
shot(89,1211,1229,(1,0),[(108,52,15,3,1),(178,85,14,-3,6),(216,97,7,4,7),(84,99,10,0,0)],.65)
shot(90,1229,1242,(-3,0),[(88,56,8,-12,-3),(135,79,10,-7,-2),(212,58,12,-3,0),(212,104,12,0,0)],.6)
gridx,gridy=np.meshgrid(np.arange(W,dtype=np.float32),np.arange(H,dtype=np.float32))
xx,yy=np.meshgrid(np.linspace(0,250,90),np.linspace(0,125,46));sample=np.c_[xx.ravel(),yy.ravel()]
def field(controls):
 pts=[];vec=[]
 def point(x,y,dx,dy):pts.append((x,y));vec.append((dx,dy))
 for x in np.linspace(-5,255,9):
  for y in (-6,131):point(x,y,0,0)
 for y in (17,42,66,92,114):
  for x in (-7,257):point(x,y,0,0)
 for x,y,r,dx,dy in controls:
  point(x,y,dx,dy)
  for ex,ey in [(1,0),(-1,0),(0,1),(0,-1)]:
   point(x+r*.78*ex,y+r*.78*ey,dx,dy)
 disp=RBFInterpolator(np.array(pts),np.array(vec),kernel='thin_plate_spline',smoothing=3)(sample).reshape(46,90,2)
 disp=np.clip(disp,[-22,-17],[22,17])
 return cv.resize(disp.astype('float32'),(W,H),interpolation=cv.INTER_CUBIC)
def fit(im):
 h,ww=im.shape[:2]; nw=min(ww,2*h); nh=nw//2
 x=(ww-nw)//2;y=(h-nh)//2
 return cv.resize(im[y:y+nh,x:x+nw],(W,H),interpolation=cv.INTER_AREA)
source=cv.VideoCapture(str(old))
output=folder/'完整素材_人物替换_打斗优化版.mp4'
tmp=w/'fight_refine_output.mp4'
proc=sp.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','30','-i','pipe:0','-i',str(old),'-map','0:v','-map','1:a:0','-c:a','copy','-c:v','libx264','-threads','6','-preset','veryfast','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)],stdin=sp.PIPE)
lookup={i:c for c in clips for i in range(c[1],c[2])}
for i in range(1373):
 ok,img=source.read();assert ok
 if i in lookup:
  n,a,b,(cx,cy),ctl,hit,_=lookup[i]
  if i==a:
   path=next((base/'完整素材_场景与人物替换'/'02_人物替换成品').glob(by[n]['prefix']+'*.png'))
   cel=fit(cv.imdecode(np.fromfile(path,np.uint8),1));f=field(ctl)
   # Pose reaches contact quickly and settles after a small recoil. The block picks up an incoming blow already in flight.
   guard=n in (34,59,71,80,81)
   tx=[0,hit*.47,hit-.045,hit,hit+.07,1.]
   vals=[-.48,-.62,.36,.46,.47,.06] if not guard else [-.18,-.24,.51,.64,.65,.08]
   traj=PchipInterpolator(tx,vals)
  p=(i-a)/(b-a-1);u=float(traj(p));z=1.045+.015*p
  # Inverse warp with a second location estimate keeps fast fists solid at the wrist.
  mx=(gridx-W*.5)/z+W*.5-f[:,:,0]*(W/250)*u-cx*(p-.45)*W/250
  my=(gridy-H*.5)/z+H*.5-f[:,:,1]*(H/125)*u-cy*(p-.45)*H/125
  img=cv.remap(cel,mx,my,cv.INTER_LINEAR,borderMode=cv.BORDER_REFLECT101)
  if 0<p<hit and abs(float(traj.derivative()(p)))/(b-a)>0.11:
   u0=float(traj(max(0,p-.35/(b-a-1))))
   blurred=cv.remap(cel,mx+f[:,:,0]*(W/250)*(u-u0),my+f[:,:,1]*(H/125)*(u-u0),cv.INTER_LINEAR,borderMode=cv.BORDER_REFLECT101)
   img=cv.addWeighted(img,.76,blurred,.24,0)
 proc.stdin.write(img.tobytes())
proc.stdin.close();assert proc.wait()==0
source.release()
import shutil
shutil.copyfile(tmp,output)
print('reanimated',len(lookup),'frames',len(clips),'shots',output)

