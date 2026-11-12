from pathlib import Path
import json,subprocess,cv2
from PIL import Image,ImageDraw
out=Path(__file__).resolve().parent
sp=out/'整片生成状态.json'
s=json.loads(sp.read_text(encoding='utf-8'))
assert s.get('gen_status')=='success','Generation must finish first'
clips=list((out/'平台原始成片').glob('*.mp4'))
assert len(clips)==1, f'Expected one clip, got {len(clips)}'
raw=clips[0]
probe=lambda p:json.loads(subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(p)],capture_output=True,text=True,encoding="utf-8",check=True).stdout)
meta=probe(raw); vs=next(x for x in meta['streams'] if x['codec_type']=='video')
duration=float(vs.get('duration') or meta['format']['duration'])
assert 7<duration<11, f'Unexpected duration {duration}; inspect manually'
final=out/'素材2_人物替换_完整9秒_原声.mp4'
filt=f'[0:v]setpts={9/duration:.12f}*(PTS-STARTPTS),fps=30,tpad=stop_mode=clone:stop_duration=0.1,trim=duration=9,setpts=PTS-STARTPTS[v];[1:a]atrim=duration=9,asetpts=PTS-STARTPTS[a]'
subprocess.run(['ffmpeg','-y','-v','error','-i',str(raw),'-i',s['source_audio'],'-filter_complex',filt,'-map','[v]','-map','[a]','-r','30','-fps_mode','cfr','-frames:v','270','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-t','9','-movflags','+faststart',str(final)],check=True)
m=probe(final); v=next(x for x in m['streams'] if x['codec_type']=='video')
assert v['avg_frame_rate']=='30/1' and int(v['nb_frames'])==270
assert abs(float(m['format']['duration'])-9)<0.02
assert any(x['codec_type']=='audio' for x in m['streams'])
subprocess.run(['ffmpeg','-v','error','-i',str(final),'-f','null','-'],check=True,capture_output=True)
cap=cv2.VideoCapture(str(final)); samples=[]; n=0
while True:
 ok,f=cap.read()
 if not ok: break
 if n%6==0: samples.append((n,cv2.cvtColor(f,cv2.COLOR_BGR2RGB)))
 n+=1
cap.release();assert n==270
for page in range(2):
 sel=samples[page*25:(page+1)*25];rows=(len(sel)+4)//5
 canvas=Image.new('RGB',(1600,rows*165),(32,32,32)); d=ImageDraw.Draw(canvas)
 for i,(idx,arr) in enumerate(sel):
  im=Image.fromarray(arr);im.thumbnail((316,139));x=(i%5)*320+(320-im.width)//2;y=(i//5)*165+22
  canvas.paste(im,(x,y));d.text(((i%5)*320+6,(i//5)*165+5),f'{idx/30:.2f}s  frame {idx}',fill='white')
 canvas.save(out/f'成片画面检查_{page+1}.jpg',quality=93)
(out/'成片媒体信息.json').write_text(json.dumps({'raw':meta,'final':m},ensure_ascii=False,indent=2),encoding='utf-8')
s.update(stage='downloaded_conformed_and_decoded',raw_file=str(raw),final_file=str(final),final_duration=9,final_fps=30,final_frames=270,width=v['width'],height=v['height'],source_video_duration=duration,normalization_speed_ratio=duration/9,has_original_audio=True,full_decode_passed=True)
sp.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:s[k] for k in ['stage','final_file','final_duration','final_fps','final_frames','width','height','source_video_duration','has_original_audio']},ensure_ascii=False))
