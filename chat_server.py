"""Chat server: 4B responds immediately, 35B verifies in background.
User sees 4B response in 10s. Verification updates the response after."""
import json, time, sys, os, re, threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from kandiga.engine import KandigaEngine
import mlx.core as mx
import socket

mx.set_cache_limit(256 * 1024 * 1024)

_stdout, _stderr = sys.stdout, sys.stderr
sys.stdout = open(os.devnull, 'w'); sys.stderr = open(os.devnull, 'w')
writer = KandigaEngine(model_path="mlx-community/Qwen3.5-4B-4bit", fast_mode=True)
writer.load()
verifier = KandigaEngine(fast_mode=True)
verifier.load()
sys.stdout, sys.stderr = _stdout, _stderr
print("Models loaded.")

# State
last_verify = {"status": "idle", "result": None}
verify_lock = threading.Lock()

def reset_sessions():
    try: writer.end_session()
    except: pass
    try: verifier.end_session()
    except: pass
    writer.start_session()
    for _ in writer.session_generate(
        "Answer concisely. Show work on calculations. Never repeat the user's input.", max_tokens=3):
        pass
    verifier.start_session()
    with verify_lock:
        last_verify["status"] = "idle"
        last_verify["result"] = None

reset_sessions()

def run_verification(question, response):
    """Run 35B verification (called from main thread AFTER 4B responds)."""
    with verify_lock:
        last_verify["status"] = "running"
        last_verify["result"] = None
    try:
        mx.clear_cache()
        result = ""
        for token in verifier.session_generate(
            f'Review this AI response. Respond ONLY in JSON:\n'
            f'{{"verified": true/false, "errors": ["error and correction"], "missing": ["missing item"], "remove": ["item to remove"]}}\n\n'
            f"Question: {question[:500]}\nResponse: {response[:600]}\n\nJSON:",
            max_tokens=150):
            result += token

        # Parse JSON
        start = result.find('{')
        end = result.rfind('}') + 1
        parsed = None
        if start >= 0 and end > start:
            try:
                parsed = json.loads(result[start:end])
            except:
                parsed = {"verified": False, "errors": [result.strip()[:100]]}
        else:
            parsed = {"verified": True, "errors": [], "missing": [], "remove": []}

        with verify_lock:
            last_verify["status"] = "done"
            last_verify["result"] = parsed
    except Exception as e:
        with verify_lock:
            last_verify["status"] = "error"
            last_verify["result"] = {"verified": False, "errors": [str(e)]}

HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Kandiga Chat</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f1117;color:#e2e4e9;font-family:-apple-system,system-ui,sans-serif;height:100vh;display:flex;flex-direction:column}
header{padding:14px 24px;border-bottom:1px solid #2a2d3a;background:#1a1d27;display:flex;justify-content:space-between;align-items:center}
h1{font-size:18px;font-weight:700;color:#4aaed9}
.hdr-right{display:flex;gap:8px;align-items:center}
.badge{font-size:11px;color:#8b8fa3;background:#2a2d3a;padding:4px 10px;border-radius:6px}
.new-chat{background:none;color:#4aaed9;border:1px solid #4aaed9;padding:6px 14px;border-radius:8px;font-size:12px;cursor:pointer;font-weight:600}
.new-chat:hover{background:#4aaed920}
#messages{flex:1;overflow-y:auto;padding:20px 24px;display:flex;flex-direction:column;gap:16px}
.msg{max-width:85%;padding:12px 16px;border-radius:12px;font-size:14px;line-height:1.7;word-wrap:break-word}
.msg h3{font-size:15px;color:#4aaed9;margin:10px 0 6px}
.msg strong{color:#e2e4e9}
.msg code{background:#2a2d3a;padding:1px 5px;border-radius:4px;font-size:13px}
.user{align-self:flex-end;background:#4aaed920;border:1px solid #4aaed940}
.ai{align-self:flex-start;background:#1a1d27;border:1px solid #2a2d3a}
.verify-badge{font-size:12px;margin-top:10px;padding:8px 12px;border-radius:8px;display:block}
.v-ok{background:#4ade8015;color:#4ade80;border:1px solid #4ade8030}
.v-warn{background:#fbbf2415;color:#fbbf24;border:1px solid #fbbf2430}
.v-checking{background:#4aaed910;color:#8b8fa3;border:1px solid #2a2d3a}
.v-item{font-size:12px;margin:3px 0;padding-left:8px;border-left:2px solid currentColor}
.meta{font-size:11px;color:#8b8fa3;margin-top:6px}
#input-area{padding:16px 24px;border-top:1px solid #2a2d3a;background:#1a1d27;display:flex;gap:10px}
#input{flex:1;background:#0f1117;border:1px solid #2a2d3a;border-radius:10px;padding:12px 16px;color:#e2e4e9;font-size:14px;font-family:inherit;outline:none;resize:none;min-height:44px;max-height:200px}
#input:focus{border-color:#4aaed9}
#send{background:#4aaed9;color:#000;border:none;border-radius:10px;padding:12px 20px;font-size:14px;font-weight:600;cursor:pointer}
#send:hover{opacity:0.9}
#send:disabled{opacity:0.4;cursor:not-allowed}
.empty{color:#8b8fa3;text-align:center;margin-top:40vh;font-size:14px}
</style>
</head>
<body>
<header>
  <h1>KANDIGA</h1>
  <div class="hdr-right">
    <button class="new-chat" onclick="newChat()">New Chat</button>
    <span class="badge">4B writer</span>
    <span class="badge">35B verifier</span>
  </div>
</header>
<div id="messages"><div class="empty">Ask anything. 4B responds instantly, 35B verifies after.</div></div>
<div id="input-area">
  <textarea id="input" placeholder="Type a message..." rows="1"></textarea>
  <button id="send" onclick="sendMsg()">Send</button>
</div>
<script>
const messages=document.getElementById('messages'),input=document.getElementById('input'),sendBtn=document.getElementById('send');
input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMsg()}});
input.addEventListener('input',()=>{input.style.height='auto';input.style.height=Math.min(input.scrollHeight,200)+'px'});

function md(s){
  if(!s)return'';
  s=s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  s=s.replace(/\\text\{([^}]*)\}/g,'$1');
  s=s.replace(/\\mathbf\{\\?\$?([\d,.]+)\}?/g,'<strong>$$$1</strong>');
  s=s.replace(/\\mathbf\{([^}]*)\}/g,'<strong>$1</strong>');
  s=s.replace(/\\times/g,'\u00D7');
  s=s.replace(/\\approx/g,'\u2248');
  s=s.replace(/\\\$/g,'$');
  s=s.replace(/\$\$([^$]+)\$\$/g,'$1');
  s=s.replace(/\$([^$]*[0-9][^$]*)\$/g,'$1');
  s=s.replace(/[*][*](.+?)[*][*]/g,'<strong>$1</strong>');
  s=s.replace(/[*](.+?)[*]/g,'<em>$1</em>');
  s=s.replace(/`(.+?)`/g,'<code>$1</code>');
  var lines=s.split(String.fromCharCode(10));
  var out=[];
  for(var i=0;i<lines.length;i++){
    var l=lines[i];
    if(l.match(/^###\s/))l='<h3>'+l.substring(4)+'</h3>';
    else if(l.match(/^##\s/))l='<h3>'+l.substring(3)+'</h3>';
    else if(l.trim()==='')l='<br>';
    out.push(l);
  }
  return out.join('<br>');
}

async function newChat(){
  await fetch('/reset',{method:'POST'});
  messages.innerHTML='<div class="empty">New conversation started.</div>';
  input.focus();
}

let currentVerifyDiv=null;

async function sendMsg(){
  const text=input.value.trim();
  if(!text)return;
  input.value='';input.style.height='auto';sendBtn.disabled=true;
  const empty=messages.querySelector('.empty');if(empty)empty.remove();

  // User message
  const u=document.createElement('div');u.className='msg user';u.textContent=text;messages.appendChild(u);

  // AI thinking
  const a=document.createElement('div');a.className='msg ai';
  a.innerHTML='<span style="color:#8b8fa3">4B thinking...</span>';
  messages.appendChild(a);messages.scrollTop=messages.scrollHeight;

  // Step 1: Get 4B response (fast)
  try{
    const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    const d=await r.json();
    let h=md(d.response);
    h+='<div class="meta">'+d.ttft+'s TTFT | '+d.tokens+' tok | '+d.toks+' tok/s</div>';
    h+='<div class="verify-badge v-checking" id="vbadge">35B verifying...</div>';
    a.innerHTML=h;
    messages.scrollTop=messages.scrollHeight;

    // Step 2: Trigger 35B verification (server runs it)
    fetch('/verify',{method:'POST'});

    // Step 3: Poll for verification result
    const poll=setInterval(async()=>{
      try{
        const vr=await fetch('/verify-status');
        const vs=await vr.json();
        if(vs.status==='done'){
          clearInterval(poll);
          const badge=document.getElementById('vbadge');
          if(badge && vs.result){
            const r=vs.result;
            if(r.verified && (!r.missing||r.missing.length===0)){
              badge.className='verify-badge v-ok';
              badge.innerHTML='\u2713 35B: VERIFIED';
            }else{
              badge.className='verify-badge v-warn';
              let items='';
              if(r.errors&&r.errors.length>0) r.errors.forEach(e=>{items+='<div class="v-item">\u2716 '+e+'</div>'});
              if(r.missing&&r.missing.length>0) r.missing.forEach(m=>{items+='<div class="v-item">\u2795 Missing: '+m+'</div>'});
              if(r.remove&&r.remove.length>0) r.remove.forEach(x=>{items+='<div class="v-item">\u2796 Remove: '+x+'</div>'});
              badge.innerHTML='\u26A0 35B Review:'+items;
            }
          }
        }else if(vs.status==='error'){
          clearInterval(poll);
          const badge=document.getElementById('vbadge');
          if(badge){badge.className='verify-badge v-warn';badge.textContent='\u26A0 Verification error';}
        }
      }catch(e){}
    },2000);

  }catch(e){a.innerHTML='<span style="color:#f87171">Error: '+e.message+'</span>'}
  sendBtn.disabled=false;input.focus();
}
</script>
</body>
</html>"""

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path=='/verify-status':
            with verify_lock:
                data={"status":last_verify["status"],"result":last_verify["result"]}
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
            return
        self.send_response(200)
        self.send_header('Content-Type','text/html')
        self.end_headers()
        self.wfile.write(HTML.encode())

    def do_POST(self):
        if self.path=='/reset':
            reset_sessions()
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
            self.wfile.write(b'{"ok":true}');return

        if self.path=='/chat':
            length=int(self.headers.get('Content-Length',0))
            body=json.loads(self.rfile.read(length))
            text=body.get('text','')

            # 4B responds (fast, return immediately)
            t0=time.time();first=None;response="";count=0
            input_start=text[:80] if len(text)>80 else text[:30]
            for token in writer.session_generate(text,max_tokens=2048):
                if first is None:first=time.time()
                response+=token;count+=1
                if count>20 and input_start and len(input_start)>20 and response.rstrip().endswith(input_start[:40]):
                    response=response[:response.rfind(input_start[:40])].rstrip();break
            ttft=round((first-t0) if first else time.time()-t0,1)
            decode=(time.time()-first) if first else 0.001
            toks=round(count/decode)

            # Store for verification
            with verify_lock:
                last_verify["status"]="pending"
                last_verify["result"]=None
                last_verify["_q"]=text
                last_verify["_r"]=response

            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
            self.wfile.write(json.dumps({
                'response':response,'ttft':str(ttft),'tokens':count,'toks':str(toks)
            }).encode())
            return

        if self.path=='/verify':
            # Run verification (blocking — client polls /verify-status)
            with verify_lock:
                q=last_verify.get("_q","")
                r=last_verify.get("_r","")
            if q and r:
                run_verification(q,r)
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
            self.wfile.write(b'{"ok":true}');return

    def log_message(self,*a):pass

print("Starting at http://localhost:8899")
server=HTTPServer(('127.0.0.1',8899),Handler)
server.socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
server.serve_forever()
