# -*- coding: utf-8 -*-
"""LIVE demo UI + REPLAY engine + static-export page.

Three artifacts share one body of animation code:
  * CSS / _VIZ / _LIGHTBOX / ANIM_JS  — the white theme + the animated handlers + the phase-paced
    replay engine (replayRun), all driven through a `fileUrl(name)` indirection.
  * PAGE         — the live server page (controls, auto-replay-by-link, picker, export).
  * static_index — a self-contained replay page (events embedded, relative media URLs, NO server/GPU).

Plain raw strings (NOT f-strings) so JS braces/backticks survive. Same-origin (live) / relative (static).
"""

CSS = r"""
:root{
  --bg:#f4f6fb;--panel:#fff;--ink:#0f172a;--muted:#64748b;--faint:#94a3b8;--line:#e6e9f0;--line2:#eef1f6;
  --accent:#4f46e5;--accent2:#6366f1;--accent-soft:#eef2ff;--ok:#0f9d58;--ok-bg:#e9f7ef;--warn:#b45309;
  --warn-bg:#fef6e7;--err:#dc2626;--err-bg:#fdecec;--agree:#0f9d58;--audio:#2563eb;--video:#ea7317;--conflict:#dc2626;
  --shadow:0 1px 2px rgba(16,24,40,.04),0 8px 24px rgba(16,24,40,.06);--shadow-lg:0 12px 40px rgba(16,24,40,.18);
}
*{box-sizing:border-box}html,body{margin:0}
body{font:14px/1.55 'Inter',-apple-system,'Segoe UI',Roboto,sans-serif;background:var(--bg);color:var(--ink);-webkit-font-smoothing:antialiased}
.app{max-width:1340px;margin:0 auto;padding:0 26px 60px}
header.top{position:sticky;top:0;z-index:30;background:rgba(244,246,251,.82);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:14px 26px;margin:0 -26px 18px;display:flex;align-items:center;gap:14px}
.brand{display:flex;align-items:center;gap:11px}
.logo{width:34px;height:34px;border-radius:9px;background:linear-gradient(135deg,var(--accent),var(--accent2));display:flex;align-items:center;justify-content:center;color:#fff;font-weight:800;font-size:16px;box-shadow:0 6px 16px rgba(79,70,229,.32)}
.brand h1{font-size:16px;margin:0;font-weight:700;letter-spacing:-.01em}.brand .sub{font-size:12px;color:var(--muted);margin-top:1px}
.spacer{flex:1}
.badge{font-size:11px;font-weight:600;color:var(--muted);background:#fff;border:1px solid var(--line);padding:4px 11px;border-radius:999px}
.scpill{font-size:12px;font-weight:600;padding:5px 12px;border-radius:999px;display:flex;align-items:center;gap:7px}
.scpill .dot{width:8px;height:8px;border-radius:50%}
.sc-ok{background:var(--ok-bg);color:var(--ok)}.sc-ok .dot{background:var(--ok)}
.sc-warn{background:var(--warn-bg);color:var(--warn)}.sc-warn .dot{background:var(--warn)}
.sc-err{background:var(--err-bg);color:var(--err)}.sc-err .dot{background:var(--err);animation:pulse 1.1s infinite}
.sc-checking{background:#eef2f7;color:var(--muted)}.sc-checking .dot{background:var(--faint);animation:pulse 1.1s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);padding:18px 20px;margin-bottom:18px}
.card h2{font-size:13px;font-weight:700;margin:0 0 12px;letter-spacing:.02em;text-transform:uppercase;color:#334155;display:flex;align-items:center;gap:8px}
.card h2 .sub{font-weight:500;text-transform:none;letter-spacing:0;color:var(--faint);font-size:11px}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
input[type=text]{flex:1;min-width:300px;background:#fff;border:1px solid var(--line);color:var(--ink);padding:12px 14px;border-radius:10px;font-size:14px;transition:.15s}
input[type=text]:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
.btn{background:var(--accent);border:0;color:#fff;font-weight:700;padding:12px 22px;border-radius:10px;cursor:pointer;font-size:14px;box-shadow:0 6px 16px rgba(79,70,229,.28);transition:.15s}
.btn:hover{background:#4338ca}.btn:disabled{background:#c7cbe0;box-shadow:none;cursor:not-allowed}
.btn.ghost{background:#fff;color:var(--accent);border:1px solid var(--line);box-shadow:none;padding:9px 14px;font-size:13px}
.btn.ghost:hover{background:var(--accent-soft)}
.filelbl{font-size:13px;color:var(--muted)}
input[type=file]{font-size:12px;color:var(--muted);max-width:210px}
.opts{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-top:12px;font-size:13px;color:var(--muted)}
.opts label{display:flex;align-items:center;gap:6px;cursor:pointer;user-select:none}
.sel{font-size:13px;padding:8px 10px;border:1px solid var(--line);border-radius:9px;background:#fff;color:var(--ink);max-width:260px}
.cachedBadge{font-size:12px;font-weight:700;color:var(--accent);background:var(--accent-soft);border:1px solid #dfe3fb;padding:5px 12px;border-radius:999px}
#status{margin:12px 2px 0;color:var(--muted);font-size:13px;min-height:18px}
.bar{height:8px;background:#eaeef5;border-radius:99px;overflow:hidden;margin:14px 0 12px;position:relative}
.bar>i{display:block;height:100%;width:0;border-radius:99px;background:linear-gradient(90deg,var(--accent),var(--accent2));transition:width .5s cubic-bezier(.4,0,.2,1)}
.bar>i::after{content:"";position:absolute;inset:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.5),transparent);animation:sheen 1.6s infinite}
@keyframes sheen{0%{transform:translateX(-100%)}100%{transform:translateX(220%)}}
.steps{display:flex;gap:6px;flex-wrap:wrap}
.step{flex:1;min-width:96px;display:flex;align-items:center;gap:8px;padding:8px 10px;border-radius:10px;background:#f7f9fc;border:1px solid var(--line2);font-size:12px;font-weight:600;color:var(--faint);transition:.25s}
.step .ico{width:18px;height:18px;border-radius:50%;border:2px solid #d3d9e6;display:flex;align-items:center;justify-content:center;font-size:10px;flex-shrink:0}
.step.active{background:var(--accent-soft);border-color:#c7caf2;color:var(--accent)}
.step.active .ico{border-color:var(--accent);box-shadow:0 0 0 4px rgba(79,70,229,.12);animation:pulse 1.2s infinite}
.step.done{background:var(--ok-bg);border-color:#cdeedd;color:var(--ok)}.step.done .ico{border-color:var(--ok);background:var(--ok);color:#fff}
.grid{display:grid;grid-template-columns:1.55fr 1fr;gap:18px}
@media(max-width:1020px){.grid{grid-template-columns:1fr}}
.featured{position:relative;border-radius:12px;overflow:hidden;background:#0b1020;border:1px solid var(--line);aspect-ratio:16/9}
.featured img{width:100%;height:100%;object-fit:contain;display:block}
.featured svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.featured .scan{position:absolute;left:0;right:0;height:3px;background:linear-gradient(90deg,transparent,var(--accent2),transparent);box-shadow:0 0 14px 3px rgba(99,102,241,.5);opacity:0}
.featured .cap{position:absolute;left:10px;bottom:10px;background:rgba(15,23,42,.78);color:#fff;font-size:11px;padding:4px 10px;border-radius:7px;backdrop-filter:blur(4px)}
.featured .idle{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#64748b;font-size:13px}
.readout{margin-top:10px;display:flex;flex-wrap:wrap;gap:6px;min-height:30px}
.tok{font-size:12px;padding:3px 9px;border-radius:7px;background:#f1f4f9;border:1px solid var(--line);color:#334155;opacity:0;transform:translateY(4px);animation:tokin .25s forwards}
.tok.num{background:#fff7ed;border-color:#fed7aa;color:#b45309;font-weight:700;font-variant-numeric:tabular-nums}
.tok.axis{background:var(--ok-bg);border-color:#bbe7cc;color:var(--ok);font-weight:700}
@keyframes tokin{to{opacity:1;transform:none}}
#frames{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px;margin-top:14px;max-height:560px;overflow:auto;padding:2px}
.fc{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#fff;opacity:0;transform:translateY(8px) scale(.98);animation:fcin .4s forwards;box-shadow:var(--shadow)}
@keyframes fcin{to{opacity:1;transform:none}}
.fc .imgwrap{position:relative;cursor:zoom-in;background:#0b1020;aspect-ratio:16/9;overflow:hidden}
.fc .imgwrap img{width:100%;height:100%;object-fit:cover;display:block;transition:transform .3s}
.fc:hover .imgwrap img{transform:scale(1.04)}
.fc .zoomhint{position:absolute;top:6px;right:6px;background:rgba(15,23,42,.7);color:#fff;font-size:10px;padding:2px 7px;border-radius:6px;opacity:0;transition:.2s}
.fc:hover .zoomhint{opacity:1}
.fc .ft{font-size:11px;color:var(--muted);padding:5px 8px 2px;font-variant-numeric:tabular-nums}
.fc .nums{padding:0 8px 7px;display:flex;flex-wrap:wrap;gap:4px;font-size:11px}
.fc .nums .n{color:var(--video);font-weight:700;font-variant-numeric:tabular-nums}
.fc .seeall{margin:0 8px 9px;font-size:11px;color:var(--accent);background:var(--accent-soft);border:1px solid #dfe3fb;padding:3px 9px;border-radius:7px;cursor:pointer;display:inline-block}
.fc .seeall:hover{background:#e3e7fc}
.fc .allbox{display:none;padding:0 8px 9px;max-height:160px;overflow:auto}.fc .allbox.open{display:block}
.fc .allbox .row{display:flex;justify-content:space-between;gap:8px;font-size:11px;padding:2px 0;border-bottom:1px solid var(--line2)}
.fc .allbox .row .c{color:var(--faint);font-variant-numeric:tabular-nums}
#audio{width:100%;margin-bottom:10px;display:none}
#transcript{max-height:300px;overflow:auto;padding-right:4px}
.tl{padding:5px 8px;border-radius:8px;cursor:pointer;font-size:13px;display:flex;gap:9px;transition:background .15s}
.tl:hover{background:#f5f7fb}.tl.active{background:var(--accent-soft);box-shadow:inset 3px 0 0 var(--accent)}
.tl b{color:var(--accent);font-variant-numeric:tabular-nums;flex-shrink:0;font-weight:600}
.caret{display:inline-block;width:2px;height:1em;background:var(--accent);margin-left:1px;vertical-align:-2px;animation:blink 1s steps(2) infinite}
@keyframes blink{50%{opacity:0}}
#vlm .v{font-size:12.5px;border-left:3px solid var(--video);padding:5px 0 5px 11px;margin:7px 0;color:#475569}
#vlm .v b{color:#9a3412}
#fusionSummary{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}
.tag{color:#fff;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:700}
.t-AGREE{background:var(--agree)}.t-AUDIO-ONLY{background:var(--audio)}.t-VIDEO-ONLY{background:var(--video)}.t-CONFLICT{background:var(--conflict)}
#fusionList{max-height:230px;overflow:auto;font-size:12px}
#fusionList .fl{padding:4px 0;border-bottom:1px solid var(--line2);display:flex;gap:8px;align-items:baseline;opacity:0;animation:tokin .3s forwards}
#sheet .big{font-size:24px;font-weight:800;letter-spacing:-.02em}
.kv{display:inline-flex;gap:6px;align-items:center;background:#f5f7fb;border:1px solid var(--line);border-radius:8px;padding:5px 11px;margin:4px 5px 0 0;font-size:12.5px}.kv b{color:var(--ink)}
#sheet .ex{margin:7px 0;padding:8px 11px;background:#f7f9fc;border-radius:9px;border:1px solid var(--line2)}
.clipbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
a.cliplink{display:inline-flex;align-items:center;gap:8px;text-decoration:none;font-weight:600;font-size:13px;color:var(--accent);background:var(--accent-soft);border:1px solid #dfe3fb;padding:9px 16px;border-radius:10px}
a.cliplink:hover{background:#e3e7fc}
.muted{color:var(--muted)}.err{color:var(--err)}
#scbanner{display:none;border-radius:12px;padding:11px 16px;margin-bottom:16px;font-size:13px;font-weight:600;align-items:center;gap:10px}
#scbanner.show{display:flex}
#scbanner.warn{background:var(--warn-bg);color:var(--warn);border:1px solid #f3e2c0}
#scbanner.err{background:var(--err-bg);color:var(--err);border:1px solid #f5c6c6}
#lb{position:fixed;inset:0;z-index:60;background:rgba(15,23,42,.86);backdrop-filter:blur(6px);display:none}
#lb.show{display:flex}
.lbwrap{flex:1;display:flex;flex-direction:column;min-width:0}
.lbtop{display:flex;align-items:center;gap:12px;padding:14px 18px;color:#fff}.lbtop .t{font-size:14px;font-weight:600}.lbtop .spacer{flex:1}
.lbbtn{background:rgba(255,255,255,.14);border:0;color:#fff;width:38px;height:38px;border-radius:9px;cursor:pointer;font-size:17px;font-weight:700}.lbbtn:hover{background:rgba(255,255,255,.26)}
.lbstageWrap{flex:1;overflow:hidden;position:relative;margin:0 12px;cursor:grab}.lbstageWrap.drag{cursor:grabbing}
.lbstage{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);transform-origin:center;will-change:transform}
.lbstage img{display:block;max-width:88vw;max-height:78vh}.lbstage svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.lbside{width:330px;flex-shrink:0;background:#fff;overflow:auto;padding:16px 18px}
@media(max-width:820px){.lbside{display:none}}
.lbside h3{font-size:13px;margin:0 0 4px}.lbside .row{display:flex;justify-content:space-between;gap:8px;font-size:12px;padding:3px 0;border-bottom:1px solid var(--line2)}
.lbside .row.num span:first-child{color:var(--video);font-weight:700}.lbside .c{color:var(--faint);font-variant-numeric:tabular-nums}
.ocbox{fill:rgba(99,102,241,.06);stroke:var(--accent2);stroke-width:2;vector-effect:non-scaling-stroke}
.ocbox.draw{animation:boxin .35s ease}
@keyframes boxin{from{stroke:#fff;fill:rgba(255,255,255,.25)}}
"""

# the visualization DOM (shared by the live page AND the static replay page)
_VIZ = r"""
<div class="grid">
  <div>
    <div class="card"><h2>👁 Reading the screen <span class="sub" id="readsub">OCR draws boxes &amp; reads each number it sees</span></h2>
      <div class="featured" id="featured"><div class="idle" id="fidle">the active frame appears here as it's read…</div>
        <img id="fimg" style="display:none"><svg id="fsvg" preserveAspectRatio="none"></svg>
        <div class="scan" id="fscan"></div><div class="cap" id="fcap" style="display:none"></div></div>
      <div class="readout" id="readout"></div><div id="frames"></div></div>
    <div class="card"><h2>🔗 Fusion <span class="sub">audio ↔ video agreement</span></h2><div id="fusionSummary"></div><div id="fusionList"></div></div>
  </div>
  <div>
    <div class="card"><h2>🎧 Transcript <span class="sub">Whisper · click a line to hear it</span></h2>
      <audio id="audio" controls preload="metadata"></audio>
      <div id="transcript"><span class="muted">— transcript types in —</span></div></div>
    <div class="card"><h2>🧠 Vision · Qwen3-VL <span class="sub">pattern/context only — never numbers</span></h2><div id="vlm"><span class="muted">— chart descriptions —</span></div></div>
    <div class="card"><h2>🎬 5-minute clip</h2><div class="clipbar" id="clipbar"><span class="muted">— saved when the crop finishes —</span></div></div>
    <div class="card"><h2>📄 Fact sheet</h2><div id="sheet" class="muted">— renders at 100% —</div></div>
    <div class="card"><h2>📥 Downloads <span class="sub">grounded summary + raw extraction</span></h2><div class="clipbar" id="downloads"><span class="muted">— available on completion —</span></div></div>
  </div>
</div>"""

_LIGHTBOX = r"""
<div id="lb"><div class="lbwrap"><div class="lbtop"><span class="t" id="lbtitle"></span><span class="spacer"></span>
  <button class="lbbtn" id="lbzo">−</button><button class="lbbtn" id="lbzi">+</button>
  <button class="lbbtn" id="lbrs" title="reset" style="font-size:13px">⤢</button><button class="lbbtn" id="lbx">✕</button></div>
  <div class="lbstageWrap" id="lbStageWrap"><div class="lbstage" id="lbStage"><img id="lbimg"><svg id="lbsvg" preserveAspectRatio="none"></svg></div></div></div>
  <div class="lbside"><h3>Full OCR <span class="muted" id="lbcount"></span></h3><div id="lbtokens"></div></div></div>"""

# shared animation handlers + the phase-paced REPLAY engine. Uses fileUrl(name) (defined per page).
ANIM_JS = r"""
var SVGNS="http://www.w3.org/2000/svg";
var STEPS=["download","crop","whisper","frames","ocr","vlm","fusion","sheet","summary"];
var LABEL={download:"Download",crop:"Crop",whisper:"Whisper",frames:"Frames",ocr:"OCR read",vlm:"Vision VLM",fusion:"Fusion",sheet:"Fact sheet",summary:"AI summary"};
function $(id){return document.getElementById(id)}
function esc(t){var d=document.createElement("div");d.textContent=(t==null?"":t);return d.innerHTML;}
function isNum(t){return /^[+\-▲▼]?\s*[\d][\d,]*(?:\.\d+)?%?$/.test((t||"").trim());}
function sleep(ms){return new Promise(function(r){setTimeout(r,ms);});}
var job=null,cursor=0,timer=null,replaying=false;
var frames={},frameOrder=[],ocrData={},segs=[];
var tq=[],tqRun=false,fq=[],fqRun=false,rq=[],rqRun=false,fcount={};
var chips={};
(function(){var s=$("steps");if(!s)return;STEPS.forEach(function(k){var c=document.createElement("div");c.className="step";c.innerHTML='<span class="ico"></span>'+LABEL[k];s.appendChild(c);chips[k]=c;});})();
function setBar(p){var b=$("barfill");if(b)b.style.width=Math.max(0,Math.min(100,p))+"%";}
function markStage(stage,state){var hit=false;for(var i=0;i<STEPS.length;i++){var k=STEPS[i];if(!chips[k])continue;
  if(k===stage){chips[k].className="step "+(state==="done"?"done":"active");chips[k].querySelector(".ico").textContent=state==="done"?"✓":"";hit=true;}
  else if(!hit){chips[k].className="step done";chips[k].querySelector(".ico").textContent="✓";}}}
function pushTranscript(ev){segs.push(ev);tq.push(ev);if(!tqRun)typeNext();}
function typeNext(){if(!tq.length){tqRun=false;return;}tqRun=true;var ev=tq.shift(),box=$("transcript");
  if(box.querySelector(".muted"))box.innerHTML="";var line=document.createElement("div");line.className="tl";
  line.dataset.start=ev.start;line.dataset.end=(ev.end!=null?ev.end:ev.start+4);
  line.innerHTML='<b>'+esc(ev.mmss)+'</b><span class="tx"></span><span class="caret"></span>';
  line.onclick=function(){var a=$("audio");if(a&&a.src){a.currentTime=parseFloat(line.dataset.start)||0;a.play();}};
  box.appendChild(line);box.scrollTop=box.scrollHeight;
  var words=(ev.text||"").split(/(\s+)/),tx=line.querySelector(".tx"),i=0,spd=tq.length>20?7:20;
  (function w(){if(i<words.length){tx.textContent+=words[i++];box.scrollTop=box.scrollHeight;setTimeout(w,spd);}
    else{var c=line.querySelector(".caret");if(c)c.remove();setTimeout(typeNext,tq.length>20?16:80);}})();}
function pushFrame(ev){frames[ev.frame]=ev;frameOrder.push(ev.frame);fq.push(ev);if(!fqRun)revealNext();}
function revealNext(){if(!fq.length){fqRun=false;return;}fqRun=true;var ev=fq.shift();
  var fc=document.createElement("div");fc.className="fc";fc.id="fc_"+cssId(ev.frame);
  fc.innerHTML='<div class="imgwrap"><img loading="lazy" src="'+fileUrl("thumbs/"+ev.name)+'"><span class="zoomhint">🔍 click to zoom</span></div><div class="ft">@'+ev.t+'s</div><div class="nums"></div>';
  fc.querySelector(".imgwrap").onclick=function(){openLightbox(ev.frame);};
  $("frames").appendChild(fc);if(ocrData[ev.frame])fillCard(ev.frame);
  setTimeout(revealNext,fq.length>30?55:180);}
function pushOCR(ev){ocrData[ev.frame]=ev;if(frames[ev.frame])fillCard(ev.frame);rq.push(ev.frame);if(!rqRun)readNext();}
function fillCard(name){var ev=ocrData[name],fc=$("fc_"+cssId(name));if(!fc||fc.dataset.filled)return;fc.dataset.filled="1";
  var toks=ev.tokens||[],nums=toks.filter(function(t){return isNum(t.t);}).slice(0,6);
  fc.querySelector(".nums").innerHTML=nums.map(function(t){return '<span class="n">'+esc(t.t)+'</span>';}).join("")||'<span class="muted" style="font-size:11px">no numbers</span>';
  var btn=document.createElement("div");btn.className="seeall";btn.textContent="See all extracted ("+toks.length+")";
  var all=document.createElement("div");all.className="allbox";
  all.innerHTML=toks.map(function(t){return '<div class="row '+(isNum(t.t)?"num":"")+'"><span>'+esc(t.t)+'</span><span class="c">'+Math.round((t.s||0)*100)+'%</span></div>';}).join("");
  btn.onclick=function(){all.classList.toggle("open");btn.textContent=(all.classList.contains("open")?"Hide raw OCR":"See all extracted ("+toks.length+")");};
  fc.appendChild(btn);fc.appendChild(all);}
function readNext(){if(!rq.length){rqRun=false;return;}rqRun=true;var name=rq.shift(),ev=ocrData[name],fr=frames[name];
  if(!ev||!fr){setTimeout(readNext,30);return;}var img=$("fimg"),svg=$("fsvg");$("fidle").style.display="none";img.style.display="block";
  $("fcap").style.display="block";$("fcap").textContent="frame @ "+ev.t+"s · reading "+(ev.tokens||[]).length+" tokens";
  img.onload=function(){var w=ev.w||img.naturalWidth||1280,h=ev.h||img.naturalHeight||720;
    svg.setAttribute("viewBox","0 0 "+w+" "+h);svg.innerHTML="";$("readout").innerHTML="";
    var scan=$("fscan");scan.style.transition="none";scan.style.top="0";scan.style.opacity="1";
    requestAnimationFrame(function(){scan.style.transition="top 1.1s linear";scan.style.top="100%";setTimeout(function(){scan.style.opacity="0";},1100);});
    var toks=ev.tokens||[],limit=rq.length>10?9:16,shown=Math.min(limit,toks.length),dly=rq.length>10?42:78,k=0;
    (function step(){if(k<shown){var t=toks[k++];
      if(t.b&&t.b.length===4){var r=document.createElementNS(SVGNS,"rect");r.setAttribute("x",t.b[0]);r.setAttribute("y",t.b[1]);
        r.setAttribute("width",Math.max(2,t.b[2]-t.b[0]));r.setAttribute("height",Math.max(2,t.b[3]-t.b[1]));r.setAttribute("class","ocbox draw");svg.appendChild(r);}
      var sp=document.createElement("span");sp.className="tok"+(isNum(t.t)?" num":"");sp.textContent=t.t;$("readout").appendChild(sp);setTimeout(step,dly);}
    else{if(ev.axis_price){var a=document.createElement("span");a.className="tok axis";a.textContent="현재가 "+ev.axis_price;$("readout").appendChild(a);}
      if(toks.length>shown){var m=document.createElement("span");m.className="tok";m.textContent="+"+(toks.length-shown)+" more";$("readout").appendChild(m);}
      for(var j=shown;j<toks.length;j++){var b=toks[j].b;if(b&&b.length===4){var rr=document.createElementNS(SVGNS,"rect");rr.setAttribute("x",b[0]);rr.setAttribute("y",b[1]);rr.setAttribute("width",Math.max(2,b[2]-b[0]));rr.setAttribute("height",Math.max(2,b[3]-b[1]));rr.setAttribute("class","ocbox");svg.appendChild(rr);}}
      setTimeout(readNext,rq.length>10?240:560);}})();};
  img.src=fileUrl("full/"+fr.full);}
function addVLM(ev){var v=$("vlm");if(v.querySelector(".muted"))v.innerHTML="";var d=document.createElement("div");d.className="v";
  d.innerHTML='<b>@'+ev.t+'s</b> '+esc(ev.desc||"")+(ev.stance?' <span class="muted">· stance: '+esc(ev.stance)+'</span>':'');v.appendChild(d);}
function addFusion(ev){var l=$("fusionList");var d=document.createElement("div");d.className="fl";
  d.innerHTML='<span class="tag t-'+esc(ev.tag)+'">'+esc(ev.tag)+'</span><b>'+esc(ev.mmss||"")+'</b> <span class="muted">'+esc(ev.kind)+' '+esc(ev.ticker||"")+'</span> '+esc((ev.detail||"").slice(0,80));
  l.appendChild(d);fcount[ev.tag]=(fcount[ev.tag]||0)+1;var h="";["AGREE","AUDIO-ONLY","VIDEO-ONLY","CONFLICT"].forEach(function(k){if(fcount[k])h+='<span class="tag t-'+k+'">'+k+' '+fcount[k]+'</span>';});$("fusionSummary").innerHTML=h;}
function showClip(ev){var bar=$("clipbar");bar.innerHTML="";
  var v=document.createElement("a");v.className="cliplink";v.href=fileUrl("clip.mp4");v.target="_blank";v.textContent="▶ Open 5-min clip";bar.appendChild(v);
  var d=document.createElement("a");d.className="cliplink";d.href=fileUrl("clip.mp4");d.setAttribute("download","moneyup_clip.mp4");d.textContent="⤓ Download";bar.appendChild(d);
  if(ev.audio){var a=$("audio");a.src=fileUrl("clip.m4a");a.style.display="block";}}
function showSheet(ev){var s=$("sheet");s.className="";
  var h='<div class="big">'+esc(ev.primary_name||"?")+' <span class="muted">('+esc(ev.primary_ticker||"?")+')</span></div>';
  if(ev.title)h+='<div class="muted" style="margin-top:2px">'+esc(ev.title)+'</div>';
  h+='<div style="margin-top:10px"><span class="kv">frames read <b>'+ev.n_frames+'</b></span><span class="kv">transcript <b>'+ev.n_segments+'</b></span><span class="kv">ex-ante calls <b>'+ev.n_exante+'</b></span><span class="kv">VLM <b>'+ev.n_vlm+'</b></span></div>';
  if(ev.fusion_summary){h+='<div style="margin-top:8px">';for(var k in ev.fusion_summary){h+='<span class="tag t-'+k+'">'+k+' '+ev.fusion_summary[k]+'</span> ';}h+='</div>';}
  (ev.exante||[]).forEach(function(c){h+='<div class="ex"><b>'+esc(c.name)+'</b> ('+esc(c.ticker)+') → <b style="color:var(--accent)">'+esc(c.direction)+'</b> @'+esc(c.mmss)+(c.stated_price?' · '+c.stated_price:'')+'<br><span class="muted">“'+esc((c.quote||"").slice(0,170))+'”</span></div>';});
  h+='<div style="margin-top:8px"><a class="cliplink" href="'+fileUrl("factsheet.md")+'" target="_blank">open full fact sheet</a></div>';s.innerHTML=h;}
function showMeta(ev){if(ev.title){var r=$("readsub");if(r)r.textContent=(ev.title||"").slice(0,70);}}
function showFiles(ev){var d=$("downloads");if(!d)return;var h="";
  if(ev.summary)h+='<a class="cliplink" href="'+fileUrl(ev.summary)+'" download="moneyup_summary_full_ko.docx">⬇ Summary (full · 한국어)</a>';
  if(ev.summary_en)h+='<a class="cliplink" href="'+fileUrl(ev.summary_en)+'" download="moneyup_summary_full_en.docx">⬇ Summary (full · English)</a>';
  if(ev.raw)h+='<a class="cliplink" href="'+fileUrl(ev.raw)+'" download="moneyup_raw_full.docx">⬇ Raw extraction (full · KO source)</a>';
  if(ev.raw_json)h+='<a class="cliplink" href="'+fileUrl(ev.raw_json)+'" download="moneyup_raw_full.json" style="background:#f5f7fb;color:var(--muted)">.json</a>';
  if(ev.comparison)h+='<a class="cliplink" href="'+fileUrl(ev.comparison)+'" download="moneyup_comparison.docx">⬇ Comparison: Whisper-only vs OCR+Whisper+VLM</a>';
  if(!ev.summary)h+='<span class="muted" style="font-size:12px;align-self:center">(summary skipped — no Gemini key)</span>';
  d.innerHTML=h||'<span class="muted">no files</span>';}
(function(){var a=$("audio");if(!a)return;a.addEventListener("timeupdate",function(){var t=a.currentTime,box=$("transcript");
  var lines=box.querySelectorAll(".tl"),cur=null;for(var i=0;i<lines.length;i++){var s=parseFloat(lines[i].dataset.start),e=parseFloat(lines[i].dataset.end);if(t>=s&&t<e)cur=lines[i];lines[i].classList.remove("active");}
  if(cur){cur.classList.add("active");var r=cur.getBoundingClientRect(),br=box.getBoundingClientRect();if(r.top<br.top||r.bottom>br.bottom)cur.scrollIntoView({block:"center",behavior:"smooth"});}});})();
var lbZ=1,lbX=0,lbY=0;
function lbApply(){$("lbStage").style.transform="translate(-50%,-50%) translate("+lbX+"px,"+lbY+"px) scale("+lbZ+")";}
function openLightbox(name){var fr=frames[name],ev=ocrData[name];if(!fr)return;lbZ=1;lbX=0;lbY=0;var img=$("lbimg"),svg=$("lbsvg");
  $("lbtitle").textContent="Frame @ "+fr.t+"s";
  img.onload=function(){var w=(ev&&ev.w)||img.naturalWidth,h=(ev&&ev.h)||img.naturalHeight;svg.setAttribute("viewBox","0 0 "+w+" "+h);svg.innerHTML="";
    var toks=(ev&&ev.tokens)||[];toks.forEach(function(t){if(t.b&&t.b.length===4){var r=document.createElementNS(SVGNS,"rect");r.setAttribute("x",t.b[0]);r.setAttribute("y",t.b[1]);r.setAttribute("width",Math.max(2,t.b[2]-t.b[0]));r.setAttribute("height",Math.max(2,t.b[3]-t.b[1]));r.setAttribute("class","ocbox");svg.appendChild(r);}});
    $("lbcount").textContent=toks.length?("· "+toks.length+" tokens"):"";
    $("lbtokens").innerHTML=toks.map(function(t){return '<div class="row '+(isNum(t.t)?"num":"")+'"><span>'+esc(t.t)+'</span><span class="c">'+Math.round((t.s||0)*100)+'%</span></div>';}).join("")||'<span class="muted">no OCR tokens</span>';lbApply();};
  img.src=fileUrl("full/"+fr.full);$("lb").classList.add("show");}
function closeLB(){$("lb").classList.remove("show");}
$("lbx").onclick=closeLB;$("lb").addEventListener("click",function(e){if(e.target.id==="lb")closeLB();});
document.addEventListener("keydown",function(e){if(e.key==="Escape")closeLB();});
$("lbzi").onclick=function(){lbZ=Math.min(8,lbZ*1.3);lbApply();};$("lbzo").onclick=function(){lbZ=Math.max(.4,lbZ/1.3);lbApply();};$("lbrs").onclick=function(){lbZ=1;lbX=0;lbY=0;lbApply();};
(function(){var w=$("lbStageWrap");if(!w)return;w.addEventListener("wheel",function(e){e.preventDefault();var f=e.deltaY<0?1.12:1/1.12;lbZ=Math.max(.4,Math.min(8,lbZ*f));lbApply();},{passive:false});
  var drag=false,sx=0,sy=0;w.addEventListener("mousedown",function(e){drag=true;sx=e.clientX-lbX;sy=e.clientY-lbY;w.classList.add("drag");});
  window.addEventListener("mousemove",function(e){if(drag){lbX=e.clientX-sx;lbY=e.clientY-sy;lbApply();}});window.addEventListener("mouseup",function(){drag=false;w.classList.remove("drag");});})();
function handle(ev){if(ev.pct!=null)setBar(ev.pct);switch(ev.type){
  case"meta":showMeta(ev);break;case"stage":markStage(ev.stage,ev.state);if(ev.msg&&$("status"))$("status").textContent="["+(LABEL[ev.stage]||ev.stage)+"] "+ev.msg;break;
  case"log":if($("status"))$("status").textContent=ev.msg;break;case"transcript":pushTranscript(ev);break;case"frame":pushFrame(ev);break;case"ocr":pushOCR(ev);break;
  case"vlm":addVLM(ev);break;case"fusion":addFusion(ev);break;case"clip":showClip(ev);break;case"sheet":showSheet(ev);break;case"files":showFiles(ev);break;
  case"done":setBar(100);STEPS.forEach(function(k){if(chips[k]){chips[k].className="step done";chips[k].querySelector(".ico").textContent="✓";}});
    if($("status"))$("status").innerHTML='<b style="color:var(--ok)">✓ complete — '+frameOrder.length+' frames read.</b>';if(typeof onComplete==="function")onComplete();break;
  case"error":if($("status"))$("status").innerHTML='<span class="err">✗ '+esc(ev.msg)+'</span>';if(typeof onComplete==="function")onComplete();break;}}
function reset(){cursor=0;frames={};frameOrder=[];ocrData={};segs=[];tq=[];fq=[];rq=[];fcount={};
  ["frames","fusionList","fusionSummary","readout"].forEach(function(id){if($(id))$(id).innerHTML="";});
  if($("transcript"))$("transcript").innerHTML='<span class="muted">— transcript types in —</span>';
  if($("vlm"))$("vlm").innerHTML='<span class="muted">— chart descriptions —</span>';
  if($("clipbar"))$("clipbar").innerHTML='<span class="muted">— clip —</span>';
  if($("downloads"))$("downloads").innerHTML='<span class="muted">— available on completion —</span>';
  if($("sheet")){$("sheet").className="muted";$("sheet").textContent="— renders at 100% —";}
  if($("fimg")){$("fimg").style.display="none";}if($("fidle"))$("fidle").style.display="flex";if($("fcap"))$("fcap").style.display="none";if($("fsvg"))$("fsvg").innerHTML="";
  STEPS.forEach(function(k){if(chips[k]){chips[k].className="step";chips[k].querySelector(".ico").textContent="";}});setBar(0);
  if($("audio")){$("audio").style.display="none";$("audio").removeAttribute("src");}}
function enc(s){return encodeURIComponent(s);}
function cssId(s){return (s||"").replace(/[^a-zA-Z0-9_-]/g,"_");}
// ---- PHASE-PACED REPLAY ENGINE: animate saved events at a ~2.5 min pace (no GPU, identical look) ----
function buildPhases(events){var phases=[],cur=null,pre=[];events.forEach(function(ev){
  if(ev.type==="done"||ev.type==="error")return;
  if(ev.type==="stage"&&ev.state==="running"){cur={stage:ev.stage,start:ev,content:[],end:null};phases.push(cur);}
  else if(ev.type==="stage"&&ev.state==="done"){if(cur&&cur.stage===ev.stage)cur.end=ev;}
  else if(cur)cur.content.push(ev);else pre.push(ev);});return{pre:pre,phases:phases};}
async function replayRun(events){replaying=true;reset();
  var BUD={download:4000,crop:2500,whisper:30000,frames:30000,ocr:44000,vlm:16000,fusion:12000,sheet:5000,summary:4000};
  var g=buildPhases(events);g.pre.forEach(handle);
  for(var pi=0;pi<g.phases.length&&replaying;pi++){var ph=g.phases[pi];handle(ph.start);ph.content.forEach(handle);
    await sleep(BUD[ph.stage]||8000);if(!replaying)return;if(ph.end)handle(ph.end);await sleep(150);}
  if(replaying)handle({type:"done",pct:100});}
"""

_LIVE_HEADER = r"""<header class="top"><div class="brand"><div class="logo">M</div>
  <div><h1>MoneyUp — Live Extraction</h1><div class="sub">watch the real pipeline read a stock video, frame by frame</div></div></div>
  <div class="spacer"></div><span class="badge">research / mock only — not a trading signal</span>
  <span id="scpill" class="scpill sc-checking"><span class="dot"></span><span id="sctext">checking GPU stack…</span></span></header>"""

_CONTROLS = r"""<div class="card">
  <div class="controls"><input id="url" type="text" placeholder="Paste a YouTube link (https://www.youtube.com/watch?v=…)">
    <span class="filelbl">or</span><input id="file" type="file" accept="video/*"><button id="go" class="btn">▶ Start extraction</button></div>
  <div class="opts">
    <label><input type="checkbox" id="fresh"> Run fresh (ignore cache)</label>
    <label><input type="checkbox" id="fast"> Fast mode (~2-4 min)</label>
    <span id="cachedBadge" class="cachedBadge" style="display:none">▶ replay (cached) · zero GPU</span>
    <span class="spacer"></span>
    <select id="runs" class="sel"><option value="">Replay a processed video…</option></select>
    <button id="replayLast" class="btn ghost">⟲ Replay last</button>
    <button id="exportBtn" class="btn ghost">⬚ Export replay</button></div>
  <div id="status" class="muted">Paste a link — if it was processed before, it INSTANTLY replays (zero GPU); otherwise it runs the real pipeline and caches it.</div>
  <div class="bar"><i id="barfill"></i></div><div class="steps" id="steps"></div></div>"""

_LIVE_JS = r"""
function fileUrl(n){return "/file?job="+job+"&name="+n;}
var onComplete=function(){$("go").disabled=false;if(timer){clearInterval(timer);timer=null;}};
function poll(){if(!job||replaying)return;fetch("/status?job="+job+"&cursor="+cursor).then(function(r){return r.json();}).then(function(d){
  if(d.events)d.events.forEach(handle);if(d.cursor!=null)cursor=d.cursor;if(d.pct!=null)setBar(d.pct);}).catch(function(){});}
function beginLive(id){replaying=false;job=id;cursor=0;reset();$("cachedBadge").style.display="none";$("go").disabled=true;timer=setInterval(poll,700);poll();}
function beginReplay(id,info){replaying=false;job=id;reset();$("cachedBadge").style.display="inline-block";$("go").disabled=true;
  $("status").innerHTML='<b style="color:var(--accent)">▶ replay (cached)</b> — '+esc((info&&info.title)||id)+' · zero GPU';
  fetch("/status?job="+id+"&cursor=0").then(function(r){return r.json();}).then(function(d){replayRun(d.events||[]).then(function(){$("go").disabled=false;});});}
function start(){var url=$("url").value.trim(),f=$("file").files[0],fresh=$("fresh").checked,fast=$("fast").checked;$("status").textContent="starting…";
  if(f){fetch("/upload?name="+enc(f.name)+(fast?"&fast=1":""),{method:"POST",body:f}).then(function(r){return r.json();}).then(function(d){
    if(d.error){$("status").innerHTML='<span class="err">'+esc(d.error)+'</span>';return;}beginLive(d.job);}).catch(function(){$("status").innerHTML='<span class="err">upload failed</span>';});return;}
  if(!url){$("status").innerHTML='<span class="err">paste a YouTube link or pick a file first</span>';return;}
  fetch("/start",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({url:url,fresh:fresh,fast:fast})}).then(function(r){return r.json();}).then(function(d){
    if(d.error){$("status").innerHTML='<span class="err">'+esc(d.error)+'</span>';return;}
    if(d.cached){beginReplay(d.job,d.info);}else{beginLive(d.job);}}).catch(function(){$("status").innerHTML='<span class="err">start failed</span>';});}
$("go").onclick=start;
$("url").addEventListener("keydown",function(e){if(e.key==="Enter")start();});
function loadRuns(){fetch("/runs").then(function(r){return r.json();}).then(function(d){var sel=$("runs");sel.innerHTML='<option value="">Replay a processed video…</option>';
  (d.runs||[]).forEach(function(r){var o=document.createElement("option");o.value=r.job_id;o.textContent=((r.title||r.video_id||r.job_id).slice(0,52))+(r.fast?" · fast":"")+" · "+(r.n_frames||0)+"f";sel.appendChild(o);});window._runs=d.runs||[];});}
$("runs").onchange=function(){var id=$("runs").value;if(!id)return;var info=(window._runs||[]).filter(function(r){return r.job_id===id;})[0];beginReplay(id,info);};
$("replayLast").onclick=function(){var rs=window._runs||[];if(!rs.length){$("status").textContent="no processed runs yet";return;}beginReplay(rs[0].job_id,rs[0]);};
$("exportBtn").onclick=function(){if(!job){$("status").textContent="run or replay something first";return;}$("status").textContent="exporting replay bundle…";
  fetch("/export?job="+job).then(function(r){return r.json();}).then(function(d){if(d.error){$("status").innerHTML='<span class="err">'+esc(d.error)+'</span>';return;}
    $("status").innerHTML='✓ Exported static replay to <b>'+esc(d.path)+'</b> — deploy that folder to Vercel.';}).catch(function(){$("status").innerHTML='<span class="err">export failed</span>';});};
function selfcheck(){fetch("/health").then(function(r){return r.json();}).then(function(d){var sc=d.selfcheck||{},pill=$("scpill"),txt=$("sctext"),ban=$("scbanner");
  if(!sc.checked){pill.className="scpill sc-checking";txt.textContent="checking GPU stack…";setTimeout(selfcheck,2000);return;}
  var wh=sc.whisper,ol=sc.ollama;
  if(wh==="wedged"||wh==="error"){pill.className="scpill sc-err";txt.textContent="GPU stack wedged";ban.className="show err";ban.innerHTML="⚠ <b>GPU/ML stack looks wedged</b> — faster_whisper "+(wh==="wedged"?"import hung":"failed")+". Reboot before a LIVE run (cached replays still work).";}
  else if(ol==="down"){pill.className="scpill sc-warn";txt.textContent="Ollama down";ban.className="show warn";ban.innerHTML="⚠ <b>Ollama not reachable</b> — Whisper OK, but the Vision (Qwen3-VL) stage will be skipped on live runs.";}
  else if(!sc.model){pill.className="scpill sc-warn";txt.textContent="Qwen model missing";ban.className="show warn";ban.innerHTML="⚠ Whisper OK, Ollama up, but the Qwen3-VL model isn't loaded — Vision may be thin.";}
  else{pill.className="scpill sc-ok";txt.textContent="GPU stack healthy";ban.className="";}}).catch(function(){setTimeout(selfcheck,2500);});}
selfcheck();loadRuns();
"""

_STATIC_JS = r"""
function fileUrl(n){return "./"+n;}
var onComplete=function(){var b=$("replayAgain");if(b)b.disabled=false;};
function go(){var b=$("replayAgain");if(b)b.disabled=true;replayRun(REPLAY_EVENTS).then(function(){if(b)b.disabled=false;});}
var rb=$("replayAgain");if(rb)rb.onclick=function(){replaying=false;setTimeout(go,60);};
go();
"""


def _doc(title, body):
    t = (title or "").replace("<", "").replace(">", "")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'><title>" + t + "</title>"
            "<style>" + CSS + "</style></head><body>" + body + "</body></html>")


PAGE = _doc("MoneyUp — Live Extraction",
            _LIVE_HEADER + "<div class='app'><div id='scbanner'></div>" + _CONTROLS + _VIZ + "</div>" +
            _LIGHTBOX + "<script>" + ANIM_JS + _LIVE_JS + "</script>")


def static_index(title, events_json):
    """Self-contained static replay page: events embedded, media via relative ./ paths, NO server/GPU."""
    t = (title or "replay").replace("<", "").replace(">", "")
    ev = (events_json or "[]").replace("</", "<\\/")            # safe to embed in a <script>
    header = ("<header class='top'><div class='brand'><div class='logo'>M</div>"
              "<div><h1>MoneyUp — Replay</h1><div class='sub'>" + t + "</div></div></div>"
              "<div class='spacer'></div><span class='badge'>research / mock only — not a trading signal</span>"
              "<button id='replayAgain' class='btn ghost'>⟲ Replay</button></header>")
    body = (header + "<div class='app'>" + _VIZ + "</div>" + _LIGHTBOX +
            "<script>var REPLAY_EVENTS=" + ev + ";\n" + ANIM_JS + _STATIC_JS + "</script>")
    return _doc("Replay — " + t, body)
