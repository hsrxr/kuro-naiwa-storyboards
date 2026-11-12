from pathlib import Path
import json,cv2,numpy as np,subprocess
from PIL import Image,ImageDraw
out=Path(__file__).resolve().parent;base=out.parent
statepath=out/'整片生成状态.json';state=json.loads(statepath.read_text(encoding='utf-8'))
final=out/'素材2_人物替换_完整9秒_原声.mp4'
mixed=out/'平台输出校准版_含原角色_仅供对照.mp4'
if not mixed.exists(): final.rename(mixed)
cap=cv2.VideoCapture(str(mixed));frames=[]
while True:
 ok,f=cap.read()
 if not ok:break
 frames.append(f)
cap.release();assert len(frames)==270
imgs={p.name[:2]:cv2.imdecode(np.fromfile(p,dtype=np.uint8),cv2.IMREAD_COLOR) for p in (base/'素材2_人物替换').glob('*.png')}
W,H=1280,720

def fit(im,zoom=1,cx=.5,cy=.5):
 h,w=im.shape[:2];cw=min(w,h*W/H)/zoom;ch=cw*H/W
 x=int(np.clip(cx*w-cw/2,0,w-cw));y=int(np.clip(cy*h-ch/2,0,h-ch));cw=int(cw);ch=int(ch)
 return cv2.resize(im[y:y+ch,x:x+cw],(W,H),interpolation=cv2.INTER_LANCZOS4)

def video(a,b,n,z0=1,z1=1,cx=.5):
 for j,t in enumerate(np.linspace(a,b-1,n)):
  lo=int(t);hi=min(lo+1,b-1);f=cv2.addWeighted(frames[lo],1-(t-lo),frames[hi],t-lo,0)
  yield fit(f,z0+(z1-z0)*j/max(1,n-1),cx)

def picture(key,n,z0=1,z1=1,cx=.5,cy=.5,dark=False,flare=False):
 for j in range(n):
  u=j/max(1,n-1);f=fit(imgs[key],z0+(z1-z0)*u,cx,cy)
  if dark:
   f=np.clip(f.astype(np.float32)*(.32+.08*u),0,255).astype(np.uint8)
  if flare:
   # Sweeping light within the video frame, keeping the reference drawing intact.
   strength=.28*(.5+.5*np.sin(u*np.pi*2))+.28*u**3
   grad=np.clip((np.arange(W,dtype=np.float32)/W-.48)/.52,0,1)[None,:,None]
   alpha=grad*strength
   f=np.clip(f*(1-alpha)+255*alpha,0,255).astype(np.uint8)
  yield f

def redflash(seq):
 for i,f in enumerate(seq):
  g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
  edge=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,7)
  mask=(edge>0)&(g>55+(i%3)*8)
  dest=np.zeros_like(f);dest[:]=(8,7,12);dest[mask]=(35,56,235)
  yield dest

white=np.full((H,W,3),246,np.uint8)
# Exact end-exclusive timeline from the original 270-frame video.
segments=[
 ('01',10,video(0,12,10),'generated 0:12'),
 ('白闪',1,iter([white]),'local flash'),
 ('02',14,video(16,26,14),'generated 16:26'),
 ('03',12,video(39,56,12),'generated 39:56'),
 ('03B',8,picture('09',8,1.22,1.32,.63,.4,dark=True),'existing 09 image; face close-up substitutes missing hand action'),
 ('04',8,video(67,87,8),'generated 67:87'),
 ('05',12,video(87,99,12),'generated 87:99'),
 ('06',36,picture('06',36,1.014,1.0,.5,.53,flare=True),'existing corrected 06 image to preserve both legs; camera pullback and light'),
 ('07A',21,video(112,125,21),'generated 112:125'),
 ('07B',13,video(125,135,13),'generated 125:135'),
 ('08',3,picture('08',3,1.0,1.025),'existing 08 image with short push'),
 ('08B',3,iter([fit(imgs['03'],1.08),fit(imgs['09'],1.3,.63,.4),white]),'three-frame local insert'),
 ('09',33,video(175,184,33,1.0,1.055,.58),'generated 175:184'),
 ('09B',4,video(179,184,4,1.12,1.25,.6),'face push substitutes unavailable forehead hand action'),
 ('10A',2,redflash(video(180,184,2,1.25,1.3,.6)),'red-black local flash'),
 ('10B',1,redflash(picture('06',1,2.2,2.2,.85,.4)),'shrine detail flash'),
 ('10C',4,redflash(picture('06',4,1.02,1.08,.5,.5)),'corrected character silhouette flash'),
 ('11',27,video(194,217,27),'generated 194:217'),
 ('10D',10,redflash(picture('06',10,1.0,1.12,.5,.48)),'corrected character silhouette flash'),
 ('12',30,picture('12',30,1.0,1.17,.52,.5,flare=True),'existing 12 image; eye push and strong light'),
 ('13',18,video(252,270,18,1.0,1.0,.5),'generated 252:270'),
]
assert sum(s[1] for s in segments)==270
silent=out/'完整9秒_无声修剪版.mp4'
args=['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','30','-i','pipe:0','-an','-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p','-r','30','-fps_mode','cfr','-frames:v','270','-movflags','+faststart',str(silent)]
proc=subprocess.Popen(args,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
manifest=[];counter=0;samples=[]
for ident,count,seq,note in segments:
 start=counter;seen=0
 for f in seq:
  assert f.shape==(H,W,3)
  if counter%6==0 or seen==0:samples.append((counter,f.copy()))
  proc.stdin.write(np.ascontiguousarray(f).tobytes());counter+=1;seen+=1
 assert seen==count,(ident,seen,count)
 manifest.append({'id':ident,'start_frame':start,'end_frame':counter,'frames':count,'source':note})
proc.stdin.close();err=proc.stderr.read().decode('utf-8',errors='replace');rc=proc.wait();assert rc==0,err
assert counter==270
subprocess.run(['ffmpeg','-y','-v','error','-i',str(silent),'-i',state['source_audio'],'-map','0:v','-map','1:a','-c:v','copy','-af','apad,atrim=duration=9,asetpts=PTS-STARTPTS','-c:a','aac','-b:a','192k','-t','9','-movflags','+faststart',str(final)],check=True)
meta=json.loads(subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(final)],capture_output=True,encoding='utf-8',check=True).stdout)
v=next(t for t in meta['streams'] if t['codec_type']=='video');assert v['avg_frame_rate']=='30/1' and int(v['nb_frames'])==270
assert abs(float(meta['format']['duration'])-9)<.001
subprocess.run(['ffmpeg','-v','error','-i',str(final),'-f','null','-'],check=True,capture_output=True)
for page in range((len(samples)+29)//30):
 sel=samples[page*30:(page+1)*30];rows=(len(sel)+4)//5
 canvas=Image.new('RGB',(1600,rows*204),(28,28,28));draw=ImageDraw.Draw(canvas)
 for j,(idx,f) in enumerate(sel):
  im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB));im.thumbnail((316,178));x=j%5*320+(320-im.width)//2;y=j//5*204+23
  canvas.paste(im,(x,y));draw.text((j%5*320+4,j//5*204+4),f'{idx} / {idx/30:.2f}s',fill='white')
 canvas.save(out/f'修正成片检查_{page+1}.jpg',quality=94)
(out/'最终剪辑索引.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(out/'最终成片媒体信息.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
state.update(stage='locally_repaired_pending_visual_check',final_file=str(final),silent_file=str(silent),mixed_comparison_file=str(mixed),width=W,height=H,final_duration=9,final_fps=30,final_frames=270,local_edit_manifest=str(out/'最终剪辑索引.json'),full_decode_passed=True,paid_tasks_this_turn=1,credits_spent_this_turn=252,remaining_credits=378,limitations=['Not a frame-perfect recreation.','Missing character actions are replaced locally with brief image camera motion and light effects.','Raw service generation contains original human characters and is retained only for comparison.'])
statepath.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'final_file':str(final),'frames':counter,'duration':meta['format']['duration'],'fps':v['avg_frame_rate'],'resolution':[W,H],'contact_pages':(len(samples)+29)//30},ensure_ascii=False))
