import { useState, useMemo, useEffect, useRef, useCallback } from "react";
const API = "http://localhost:8000";

const DARK={bg:"#0a0e17",bgCard:"#111827",bgHover:"#1a2332",bgModal:"#0d1117",border:"#1e293b",borderLight:"#334155",text:"#f1f5f9",textDim:"#94a3b8",textMuted:"#64748b",accent:"#3b82f6",long:"#22c55e",longBg:"rgba(34,197,94,0.12)",short:"#ef4444",shortBg:"rgba(239,68,68,0.12)",gold:"#eab308",goldBg:"rgba(234,179,8,0.12)",purple:"#a855f7",chartBg:"#0a0e17",chartGrid:"#1e293b",chartText:"#64748b"};
const LIGHT={bg:"#f8fafc",bgCard:"#ffffff",bgHover:"#f1f5f9",bgModal:"#f1f5f9",border:"#e2e8f0",borderLight:"#cbd5e1",text:"#0f172a",textDim:"#64748b",textMuted:"#94a3b8",accent:"#3b82f6",long:"#16a34a",longBg:"rgba(22,163,74,0.08)",short:"#dc2626",shortBg:"rgba(220,38,38,0.08)",gold:"#ca8a04",goldBg:"rgba(202,138,4,0.08)",purple:"#7c3aed",chartBg:"#ffffff",chartGrid:"#f1f5f9",chartText:"#64748b"};
type Th=typeof DARK;

// ── Bean Logo (suited, glasses, coffee, coins — inspired by user's image) ──
function BeanLogo({size=36}:{size?:number}){return(
<svg width={size} height={size} viewBox="0 0 120 120" fill="none">
  {/* Coin stack */}
  <ellipse cx="60" cy="105" rx="30" ry="6" fill="#ca8a04" opacity="0.3"/>
  <ellipse cx="60" cy="100" rx="22" ry="5" fill="#eab308"/><ellipse cx="60" cy="99" rx="22" ry="5" fill="#facc15"/>
  <ellipse cx="60" cy="95" rx="20" ry="4.5" fill="#eab308"/><ellipse cx="60" cy="94" rx="20" ry="4.5" fill="#facc15"/>
  {/* Body */}
  <ellipse cx="60" cy="62" rx="28" ry="34" fill="#92643a"/>
  <ellipse cx="60" cy="60" rx="25" ry="31" fill="#a8764a"/>
  {/* Suit jacket */}
  <path d="M38 58C38 58 42 82 60 85C78 82 82 58 82 58L78 55C78 55 70 62 60 62C50 62 42 55 42 55L38 58Z" fill="#1e293b"/>
  {/* White shirt V */}
  <path d="M52 55L60 70L68 55" stroke="white" strokeWidth="3" fill="none"/>
  {/* Green tie */}
  <path d="M60 58L57 68L60 80L63 68L60 58Z" fill="#22c55e"/>
  {/* Bean line */}
  <path d="M60 30C60 30 53 48 53 60C53 72 60 90 60 90" stroke="#7a5230" strokeWidth="2" strokeLinecap="round" opacity="0.3"/>
  {/* Glasses */}
  <rect x="42" y="42" width="14" height="11" rx="5" stroke="#1e293b" strokeWidth="2.5" fill="none"/>
  <rect x="64" y="42" width="14" height="11" rx="5" stroke="#1e293b" strokeWidth="2.5" fill="none"/>
  <line x1="56" y1="47" x2="64" y2="47" stroke="#1e293b" strokeWidth="2"/>
  {/* Eyes */}
  <circle cx="49" cy="47" r="3" fill="#1e293b"/><circle cx="71" cy="47" r="3" fill="#1e293b"/>
  <circle cx="48" cy="46" r="1.2" fill="white"/><circle cx="70" cy="46" r="1.2" fill="white"/>
  {/* Blush */}
  <ellipse cx="39" cy="54" rx="4" ry="2.5" fill="#e8a0a0" opacity="0.4"/>
  <ellipse cx="81" cy="54" rx="4" ry="2.5" fill="#e8a0a0" opacity="0.4"/>
  {/* Smile */}
  <path d="M50 57Q60 65 70 57" stroke="#5c3a1e" strokeWidth="2" strokeLinecap="round" fill="none"/>
  {/* Hair tuft */}
  <path d="M55 28C55 28 58 22 60 24C62 22 65 28 65 28" stroke="#7a5230" strokeWidth="2.5" strokeLinecap="round" fill="#92643a"/>
  {/* Left arm — coffee */}
  <path d="M34 60C28 56 22 50 26 46" stroke="#a8764a" strokeWidth="4" strokeLinecap="round"/>
  <rect x="18" y="38" width="12" height="14" rx="3" fill="#78716c"/><rect x="18" y="38" width="12" height="3" rx="1.5" fill="#57534e"/>
  <path d="M22 36C23 33 25 33 26 36" stroke="#94a3b8" strokeWidth="1.5" strokeLinecap="round" opacity="0.5"/>
  <path d="M25 34C26 31 28 32 28 35" stroke="#94a3b8" strokeWidth="1" strokeLinecap="round" opacity="0.4"/>
  {/* Right arm — thumbs up */}
  <path d="M86 60C92 56 96 52 94 48" stroke="#a8764a" strokeWidth="4" strokeLinecap="round"/>
  <circle cx="94" cy="44" r="5" fill="#a8764a"/>
  <path d="M94 39L94 34" stroke="#a8764a" strokeWidth="3.5" strokeLinecap="round"/>
  {/* Chart arrow */}
  <path d="M88 22L96 14L104 18" stroke="#22c55e" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"/>
  <polygon points="104,14 108,18 104,22" fill="#22c55e"/>
  {/* Sparkles */}
  <circle cx="14" cy="30" r="1.5" fill="#eab308" opacity="0.6"/><circle cx="106" cy="30" r="1" fill="#22c55e" opacity="0.6"/>
  <circle cx="20" cy="20" r="1" fill="#3b82f6" opacity="0.5"/>
</svg>
)}

// ── Loading Spinner with animation ──
function LoadingBean({t,msg="Loading..."}:{t:Th;msg?:string}){
  return(
    <div style={{textAlign:"center",padding:60}}>
      <style>{`
        @keyframes beanBounce{0%,100%{transform:translateY(0) rotate(0deg)}25%{transform:translateY(-12px) rotate(-5deg)}75%{transform:translateY(-6px) rotate(5deg)}}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
        @keyframes dots{0%{content:""}33%{content:"."}66%{content:".."}100%{content:"..."}}
        .bean-bounce{animation:beanBounce 1.2s ease-in-out infinite}
        .pulse-text{animation:pulse 1.5s ease-in-out infinite}
      `}</style>
      <div className="bean-bounce" style={{display:"inline-block"}}><BeanLogo size={56}/></div>
      <div className="pulse-text" style={{fontSize:15,fontWeight:700,marginTop:16,color:t.text}}>{msg}</div>
      <div style={{display:"flex",justifyContent:"center",gap:4,marginTop:10}}>
        {[0,1,2,3].map(i=><div key={i} style={{width:6,height:6,borderRadius:"50%",background:t.accent,opacity:0.3,animation:`pulse 1.2s ease-in-out ${i*0.2}s infinite`}}/>)}
      </div>
    </div>
  );
}

// ── Pill / Badge helpers ──
function Pill({active,children,onClick,t,s}:{active:boolean;children:React.ReactNode;onClick:()=>void;t:Th;s?:boolean}){
  return <button onClick={onClick} style={{fontSize:s?10:12,fontWeight:active?700:500,padding:s?"4px 9px":"5px 13px",borderRadius:6,cursor:"pointer",border:"none",background:active?t.accent:"transparent",color:active?"#fff":t.textDim,fontFamily:"'Outfit',sans-serif",transition:"all .15s"}}>{children}</button>;
}

const CAT_C:{[k:string]:{c:string;l:string}}={classical:{c:"#3b82f6",l:"Classical"},candlestick:{c:"#f59e0b",l:"Candle"},smb_scalp:{c:"#a855f7",l:"SMB"},quant:{c:"#10b981",l:"Quant"}};
function CatBadge({cat}:{cat:string}){const c=CAT_C[cat]||{c:"#888",l:cat};return <span style={{fontSize:10,fontWeight:700,padding:"2px 6px",borderRadius:4,background:c.c+"18",color:c.c,textTransform:"uppercase",letterSpacing:.3}}>{c.l}</span>;}
function TfBadge({tf,t}:{tf:string;t:Th}){const m=tf.includes("&");return <span style={{fontSize:10,fontWeight:700,padding:"2px 6px",borderRadius:4,background:m?t.accent+"20":t.textDim+"15",color:m?t.accent:t.textDim}}>{m?"★ ":""}{tf}</span>;}
function VerdictBadge({v,t}:{v:any;t:Th}){if(!v||v.verdict==="PENDING")return null;const m:{[k:string]:{bg:string;c:string;i:string}}={CONFIRMED:{bg:t.long+"20",c:t.long,i:"✓"},CAUTION:{bg:t.gold+"20",c:t.gold,i:"⚠"},DENIED:{bg:t.short+"20",c:t.short,i:"✗"}};const c=m[v.verdict]||m.CAUTION;return <span title={v.reasoning} style={{fontSize:10,fontWeight:800,padding:"2px 8px",borderRadius:4,background:c.bg,color:c.c,border:`1px solid ${c.c}30`,cursor:"help"}}>{c.i} {v.verdict}</span>;}
function ScoreCell({score,t}:{score:number;t:Th}){const c=score>=65?t.long:score>=45?t.gold:t.short;return <span style={{fontFamily:"'JetBrains Mono',monospace",fontSize:14,fontWeight:800,color:c,padding:"3px 7px",borderRadius:5,background:c+"15",display:"inline-block",minWidth:38,textAlign:"center"}}>{score.toFixed(0)}</span>;}

// ── Correlation Label ──
function CorrLabel({corr,t}:{corr:any;t:Th}){
  if(!corr||!corr.label)return null;
  const color = corr.color || t.textDim;
  return(
    <span title={`SPY r=${corr.pearson_r} | ${corr.direction_agreement}% same dir | Score: ${corr.correlation_score}`} style={{
      fontSize:10,fontWeight:700,padding:"2px 7px",borderRadius:4,
      background:color+"18",color,cursor:"help",border:`1px solid ${color}25`,
    }}>
      {corr.label} <span style={{opacity:0.6,fontSize:9}}>({corr.grade})</span>
    </span>
  );
}

// ── Regime Pill ──
const REG:{[k:string]:{l:string;i:string;ck:keyof Th}}={trending_bull:{l:"TRENDING BULL",i:"▲",ck:"long"},trending_bear:{l:"TRENDING BEAR",i:"▼",ck:"short"},high_volatility:{l:"HIGH VOLATILITY",i:"◆",ck:"gold"},mean_reverting:{l:"MEAN REVERTING",i:"◎",ck:"accent"}};
function RegimePill({r,t}:{r:any;t:Th}){if(!r?.regime)return null;const c=REG[r.regime];if(!c)return null;const co=t[c.ck]as string;return <span style={{fontSize:10,fontWeight:700,padding:"3px 10px",borderRadius:6,background:co+"15",color:co,border:`1px solid ${co}30`}}>{c.i} {c.l}</span>;}

// ── Backtest Modal ──
const PAT_CAT:{[k:string]:string}={"Head & Shoulders":"classical","Inverse H&S":"classical","Double Top":"classical","Double Bottom":"classical","Triple Top":"classical","Triple Bottom":"classical","Ascending Triangle":"classical","Descending Triangle":"classical","Symmetrical Triangle":"classical","Bull Flag":"classical","Bear Flag":"classical","Pennant":"classical","Cup & Handle":"classical","Rectangle":"classical","Rising Wedge":"classical","Falling Wedge":"classical","Bullish Engulfing":"candlestick","Bearish Engulfing":"candlestick","Morning Star":"candlestick","Evening Star":"candlestick","Hammer":"candlestick","Shooting Star":"candlestick","Doji":"candlestick","Dragonfly Doji":"candlestick","Three White Soldiers":"candlestick","Three Black Crows":"candlestick","RubberBand Scalp":"smb_scalp","HitchHiker Scalp":"smb_scalp","ORB 15min":"smb_scalp","ORB 30min":"smb_scalp","Second Chance Scalp":"smb_scalp","BackSide Scalp":"smb_scalp","Fashionably Late":"smb_scalp","Spencer Scalp":"smb_scalp","Gap Give & Go":"smb_scalp","Tidal Wave":"smb_scalp","Breaking News":"smb_scalp","Momentum Breakout":"quant","Vol Compression Breakout":"quant","Mean Reversion":"quant","Trend Pullback":"quant","Gap Fade":"quant","Relative Strength Break":"quant","Range Expansion":"quant","Volume Breakout":"quant","VWAP Reversion":"quant","Donchian Breakout":"quant"};

function BacktestModal({open,onClose,t}:{open:boolean;onClose:()=>void;t:Th}){
  const[data,setData]=useState<any>(null);const[sort,setSort]=useState("edge_score");const[loading,setLoading]=useState(true);
  useEffect(()=>{if(!open)return;setLoading(true);fetch(`${API}/api/backtest/patterns?sort=${sort}`).then(r=>r.json()).then(d=>{setData(d);setLoading(false)}).catch(()=>setLoading(false))},[open,sort]);
  if(!open)return null;const summary=data?.summary||{};const patterns=data?.patterns||[];
  return(
    <div style={{position:"fixed",inset:0,zIndex:1000,display:"flex",alignItems:"center",justifyContent:"center",background:"rgba(0,0,0,0.75)",backdropFilter:"blur(6px)"}} onClick={onClose}>
      <div onClick={e=>e.stopPropagation()} style={{background:t.bgModal,border:`1px solid ${t.border}`,borderRadius:16,width:"94vw",maxWidth:1300,maxHeight:"88vh",overflow:"hidden",display:"flex",flexDirection:"column",boxShadow:"0 25px 50px rgba(0,0,0,0.5)"}}>
        <div style={{padding:"16px 24px",borderBottom:`1px solid ${t.border}`,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
          <div><span style={{fontSize:20,fontWeight:800,color:t.text}}>📊 Backtest Results</span>
            {summary.total_signals&&<span style={{fontSize:13,color:t.textDim,marginLeft:12}}>{summary.total_symbols} symbols • {summary.total_signals?.toLocaleString()} signals • {summary.overall_win_rate}% WR</span>}
          </div>
          <div style={{display:"flex",gap:6,alignItems:"center"}}>
            <span style={{fontSize:11,color:t.textMuted}}>Sort:</span>
            {["edge_score","win_rate","profit_factor","expectancy","total_signals"].map(s=>{
              const labels:{[k:string]:string}={edge_score:"Edge",win_rate:"WR%",profit_factor:"PF",expectancy:"Exp",total_signals:"Signals"};
              return <Pill key={s} active={sort===s} onClick={()=>setSort(s)} t={t} s>{labels[s]}</Pill>;
            })}
            <button onClick={onClose} style={{background:"none",border:"none",color:t.textDim,fontSize:20,cursor:"pointer",marginLeft:10}}>✕</button>
          </div>
        </div>
        <div style={{overflow:"auto",flex:1,padding:"0 8px"}}>
          {loading?<LoadingBean t={t} msg="Loading backtest data..."/>:(
            <table style={{width:"100%",borderCollapse:"collapse",fontSize:13,fontFamily:"'JetBrains Mono',monospace"}}>
              <thead><tr style={{position:"sticky",top:0,background:t.bgModal,borderBottom:`2px solid ${t.borderLight}`,fontSize:11,fontWeight:700,color:t.textMuted,textTransform:"uppercase",letterSpacing:.5}}>
                {["Pattern","Cat","Signals","Win%","PF","Exp/R","Avg Win","Avg Loss","Edge","Grade"].map(h=><th key={h} style={{padding:"10px 8px",textAlign:h==="Pattern"?"left":"right"}}>{h}</th>)}
              </tr></thead>
              <tbody>{patterns.map((p:any)=>{
                const grade=p.edge_score>=70&&p.total_signals>=20?"A":p.edge_score>=55&&p.total_signals>=10?"B":p.edge_score>=40&&p.total_signals>=5?"C":p.edge_score>=25?"D":"F";
                const gc:{[k:string]:string}={A:t.long,B:t.accent,C:t.gold,D:t.short,F:t.short};const gc2=gc[grade]||t.textDim;
                const catKey=PAT_CAT[p.name]||"";
                return(<tr key={p.name} style={{borderBottom:`1px solid ${t.border}`}}>
                  <td style={{padding:"9px 8px",fontWeight:700,color:t.text,fontFamily:"'Outfit',sans-serif",fontSize:14}}>{p.name}</td>
                  <td style={{textAlign:"right"}}><CatBadge cat={catKey}/></td>
                  <td style={{textAlign:"right",color:t.textDim}}>{p.total_signals?.toLocaleString()}</td>
                  <td style={{textAlign:"right",color:p.win_rate>=55?t.long:p.win_rate>=45?t.gold:t.short,fontWeight:700}}>{p.win_rate?.toFixed(1)}%</td>
                  <td style={{textAlign:"right",color:p.profit_factor>=2?t.long:p.profit_factor>=1?t.text:t.short}}>{p.profit_factor?.toFixed(1)}</td>
                  <td style={{textAlign:"right",color:p.expectancy>=0?t.long:t.short}}>{p.expectancy>=0?"+":""}{p.expectancy?.toFixed(3)}</td>
                  <td style={{textAlign:"right",color:t.long}}>{p.avg_win_r?.toFixed(1)}R</td>
                  <td style={{textAlign:"right",color:t.short}}>{p.avg_loss_r?.toFixed(1)}R</td>
                  <td style={{textAlign:"right"}}><span style={{padding:"2px 7px",borderRadius:4,background:gc2+"18",color:gc2,fontWeight:800}}>{p.edge_score?.toFixed(0)}</span></td>
                  <td style={{textAlign:"right"}}><span style={{fontWeight:800,color:gc2,fontSize:15}}>{grade}</span></td>
                </tr>);
              })}</tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Chart ──
function TradeChart({setup,onClose,t}:{setup:any;onClose:()=>void;t:Th}){
  const ref=useRef<HTMLDivElement>(null);const[loading,setLoading]=useState(true);const[err,setErr]=useState("");
  useEffect(()=>{if(!ref.current)return;let ch:any=null;
    (async()=>{try{const lc=await import("lightweight-charts");const tf=setup.timeframe_detected?.includes("15m")?"15min":"5min";const r=await fetch(`${API}/api/chart/${setup.symbol}?timeframe=${tf}&days_back=5`);const d=await r.json();if(d.error)throw new Error(d.error);if(!d.bars?.length)throw new Error("No data");if(ref.current)ref.current.innerHTML="";
      ch=lc.createChart(ref.current!,{width:ref.current!.clientWidth,height:380,layout:{background:{color:t.chartBg}as any,textColor:t.chartText,fontFamily:"JetBrains Mono,monospace"},grid:{vertLines:{color:t.chartGrid},horzLines:{color:t.chartGrid}},crosshair:{mode:lc.CrosshairMode.Normal},rightPriceScale:{borderColor:t.border},timeScale:{borderColor:t.border,timeVisible:true}});
      let s:any;if((lc as any).CandlestickSeries)s=ch.addSeries((lc as any).CandlestickSeries,{upColor:t.long,downColor:t.short,borderUpColor:t.long,borderDownColor:t.short,wickUpColor:t.long,wickDownColor:t.short});else s=ch.addCandlestickSeries({upColor:t.long,downColor:t.short,borderUpColor:t.long,borderDownColor:t.short,wickUpColor:t.long,wickDownColor:t.short});
      s.setData(d.bars);
      s.createPriceLine({price:setup.entry_price,color:t.accent,lineWidth:2,lineStyle:lc.LineStyle.Dashed,axisLabelVisible:true,title:"ENTRY"});
      s.createPriceLine({price:setup.stop_loss,color:t.short,lineWidth:2,lineStyle:lc.LineStyle.Dashed,axisLabelVisible:true,title:"STOP"});
      s.createPriceLine({price:setup.target_price,color:t.long,lineWidth:2,lineStyle:lc.LineStyle.Dashed,axisLabelVisible:true,title:"TARGET"});
      Object.entries(setup.key_levels||{}).forEach(([n,p])=>{if(typeof p==="number"&&p>0)s.createPriceLine({price:p,color:t.textMuted,lineWidth:1,lineStyle:lc.LineStyle.Dotted,axisLabelVisible:false,title:n})});
      ch.timeScale().fitContent();setLoading(false);
      const ro=new ResizeObserver(()=>{if(ref.current&&ch)ch.applyOptions({width:ref.current.clientWidth})});ro.observe(ref.current!);
    }catch(e:any){setErr(e.message);setLoading(false)}})();return()=>{if(ch)ch.remove()};},[setup,t]);
  return(<div style={{background:t.bg,border:`1px solid ${t.border}`,borderRadius:12,padding:14,margin:"8px 0"}}>
    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8}}>
      <span style={{fontSize:15,fontWeight:700,color:t.text}}>{setup.symbol} — {setup.pattern_name} ({setup.timeframe_detected})</span>
      <div style={{display:"flex",gap:12,fontSize:13,fontWeight:700,alignItems:"center"}}>
        <span style={{color:t.accent}}>Entry ${setup.entry_price?.toFixed(2)}</span>
        <span style={{color:t.short}}>Stop ${setup.stop_loss?.toFixed(2)}</span>
        <span style={{color:t.long}}>Target ${setup.target_price?.toFixed(2)}</span>
        <button onClick={onClose} style={{background:"none",border:"none",cursor:"pointer",color:t.textDim,fontSize:16}}>✕</button>
      </div>
    </div>
    {setup.ai_verdict?.reasoning&&setup.ai_verdict.verdict!=="PENDING"&&(
      <div style={{fontSize:13,color:t.textDim,padding:"8px 12px",background:t.bgCard,borderRadius:8,marginBottom:8,border:`1px solid ${t.border}`}}>
        <span style={{fontWeight:700,color:t.text}}>AI: </span>{setup.ai_verdict.reasoning}
      </div>
    )}
    {loading&&<div style={{textAlign:"center",padding:30,color:t.textDim,fontSize:14}}>Loading chart...</div>}
    {err&&<div style={{textAlign:"center",padding:14,color:t.short,fontSize:12}}>Chart: {err}</div>}
    <div ref={ref} style={{width:"100%",minHeight:loading?0:380,marginTop:8}}/>
  </div>);
}

// ── Setup Row ──
function SetupRow({s,open,toggle,t,onTrack}:{s:any;open:boolean;toggle:()=>void;t:Th;onTrack:(s:any)=>void}){
  const isL=s.bias==="long";
  // Trigger text: "If AAPL hits $247.86 → BUY" or "If AAPL breaks below $398 → SHORT"
  const trigger=isL
    ?`If ${s.symbol} hits $${s.entry_price?.toFixed(2)} → BUY`
    :`If ${s.symbol} breaks $${s.entry_price?.toFixed(2)} → SHORT`;
  const det=new Date(s.detected_at);const isToday=new Date().toDateString()===det.toDateString();
  const dateStr=isToday?"Today":det.toLocaleDateString([],{month:"short",day:"numeric"});
  const timeStr=det.toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"});

  return(<div>
    <div style={{display:"flex",alignItems:"center",gap:8,padding:"10px 14px",borderBottom:`1px solid ${t.border}`,cursor:"pointer",background:open?t.bgHover:"transparent",transition:"background .15s"}}>
      <div onClick={toggle} style={{display:"flex",alignItems:"center",gap:8,flex:1,minWidth:0}}>
        <div style={{width:65}}>
          <div style={{fontSize:16,fontWeight:800,color:t.text}}>{s.symbol}</div>
          <span style={{fontSize:10,fontWeight:700,padding:"2px 7px",borderRadius:4,background:isL?t.longBg:t.shortBg,color:isL?t.long:t.short}}>{s.bias?.toUpperCase()}</span>
        </div>
        <div style={{flex:1,minWidth:140}}>
          <div style={{display:"flex",alignItems:"center",gap:5,flexWrap:"wrap"}}>
            <span style={{fontSize:14,fontWeight:700,color:t.text}}>{s.pattern_name}</span>
            <CatBadge cat={s.category}/><TfBadge tf={s.timeframe_detected} t={t}/>
            <VerdictBadge v={s.ai_verdict} t={t}/>
            <CorrLabel corr={s.spy_correlation} t={t}/>
          </div>
          <div style={{fontSize:12,color:t.accent,fontWeight:600,marginTop:3}}>{trigger}</div>
          <div style={{fontSize:11,color:t.textMuted,marginTop:1}}>Detected: {dateStr} {timeStr} • Target: ${s.target_price?.toFixed(2)} • R:R {s.risk_reward_ratio?.toFixed(1)}</div>
        </div>
        <div style={{display:"flex",gap:10}}>
          {[{l:"ENTRY",v:s.entry_price,c:t.text},{l:"STOP",v:s.stop_loss,c:t.short},{l:"TARGET",v:s.target_price,c:t.long}].map(({l,v,c})=>(
            <div key={l} style={{textAlign:"right",minWidth:55}}>
              <div style={{fontSize:9,color:t.textMuted,fontWeight:700,letterSpacing:.4}}>{l}</div>
              <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:14,fontWeight:700,color:c}}>${v?.toFixed(2)}</div>
            </div>
          ))}
        </div>
        <ScoreCell score={s.composite_score||0} t={t}/>
        <div style={{width:14,textAlign:"center",fontSize:12,color:t.textDim}}>{open?"▴":"▾"}</div>
      </div>
      <button onClick={e=>{e.stopPropagation();onTrack(s)}} title="Track this trade live" style={{
        background:t.accent+"18",border:`1px solid ${t.accent}35`,borderRadius:6,padding:"5px 10px",
        cursor:"pointer",fontSize:11,fontWeight:700,color:t.accent,whiteSpace:"nowrap",
      }}>+ Track</button>
    </div>
    {open&&<TradeChart setup={s} onClose={toggle} t={t}/>}
  </div>);
}

// ── Tracked Trade ──
function TrackedTrade({trade,price,t,onRemove}:{trade:any;price:any;t:Th;onRemove:()=>void}){
  const isL=trade.bias==="long";const entry=trade.entry_price;const stop=trade.stop_loss;const target=trade.target_price;
  const cur=price?.price||0;const risk=Math.abs(entry-stop);
  const pnlR=risk>0?(isL?(cur-entry)/risk:(entry-cur)/risk):0;
  const hitT=isL?cur>=target:cur<=target;const hitS=isL?cur<=stop:cur>=stop;
  const status=hitT?"🎯 TARGET":hitS?"🛑 STOPPED":"⏳ ACTIVE";
  const sc=hitT?t.long:hitS?t.short:t.accent;
  return(
    <div style={{display:"flex",alignItems:"center",gap:10,padding:"8px 14px",borderBottom:`1px solid ${t.border}`,fontSize:13}}>
      <span style={{fontWeight:800,color:t.text,width:55}}>{trade.symbol}</span>
      <span style={{fontSize:10,padding:"2px 6px",borderRadius:4,background:isL?t.longBg:t.shortBg,color:isL?t.long:t.short,fontWeight:700}}>{trade.bias?.toUpperCase()}</span>
      <span style={{color:t.textDim,flex:1}}>{trade.pattern_name}</span>
      <span style={{fontFamily:"'JetBrains Mono',monospace"}}>
        <span style={{color:t.textDim}}>In: ${entry?.toFixed(2)}</span>
        <span style={{color:t.text,marginLeft:10}}>Now: ${cur?cur.toFixed(2):"..."}</span>
        <span style={{color:pnlR>=0?t.long:t.short,marginLeft:10,fontWeight:800}}>{pnlR>=0?"+":""}{pnlR.toFixed(2)}R</span>
      </span>
      <span style={{fontSize:11,fontWeight:700,padding:"3px 8px",borderRadius:5,background:sc+"18",color:sc}}>{status}</span>
      <button onClick={onRemove} style={{background:"none",border:"none",color:t.textDim,cursor:"pointer",fontSize:14}}>✕</button>
    </div>
  );
}

// ── In-Play Card ──
function InPlayCard({stock,onClick,t}:{stock:any;onClick:()=>void;t:Th}){
  return(<div onClick={onClick} style={{padding:"10px 14px",background:t.bgCard,borderRadius:10,border:`1px solid ${t.border}`,cursor:"pointer",minWidth:130,flex:"0 0 auto",transition:"all .2s"}}
    onMouseEnter={e=>{e.currentTarget.style.borderColor=t.accent;e.currentTarget.style.transform="translateY(-2px)"}}
    onMouseLeave={e=>{e.currentTarget.style.borderColor=t.border;e.currentTarget.style.transform="translateY(0)"}}>
    <span style={{fontSize:16,fontWeight:800,color:t.text}}>{stock.symbol}</span>
    <div style={{fontSize:11,color:t.textDim,marginTop:3,lineHeight:1.3}}>{stock.reason?.slice(0,50)}</div>
  </div>);
}

// ── Main App ──
export default function App(){
  const[dark,setDark]=useState(true);const t=dark?DARK:LIGHT;
  const[view,setView]=useState<"opp"|"scan">("opp");
  const[symbol,setSymbol]=useState("AAPL");const[scanSetups,setScanSetups]=useState<any[]>([]);
  const[topSetups,setTopSetups]=useState<any[]>([]);const[inPlay,setInPlay]=useState<any[]>([]);
  const[mktSummary,setMktSummary]=useState("");
  const[loading,setLoading]=useState(false);const[topLoading,setTopLoading]=useState(true);
  const[error,setError]=useState("");const[chartIdx,setChartIdx]=useState<number|null>(null);
  const[fBias,setFBias]=useState("ALL");const[fCat,setFCat]=useState("ALL");const[sortBy,setSortBy]=useState<"score"|"rr">("score");
  const[mode,setMode]=useState<"today"|"active">("today");
  const[regime,setRegime]=useState<any>(null);const[hotStrats,setHotStrats]=useState<any[]>([]);
  const[pc,setPc]=useState(47);const[mktOpen,setMktOpen]=useState(true);
  const[btOpen,setBtOpen]=useState(false);
  const[tracked,setTracked]=useState<any[]>([]);const[prices,setPrices]=useState<any>({});

  useEffect(()=>{
    fetch(`${API}/api/health`).then(r=>r.json()).then(d=>{if(d.patterns)setPc(d.patterns);if(d.market_open!==undefined)setMktOpen(d.market_open)}).catch(()=>{});
    fetch(`${API}/api/regime`).then(r=>r.json()).then(d=>{if(d.regime)setRegime(d)}).catch(()=>{});
    fetch(`${API}/api/hot-strategies?top_n=5`).then(r=>r.json()).then(d=>{if(d.strategies)setHotStrats(d.strategies)}).catch(()=>{});
    setTopLoading(true);
    fetch(`${API}/api/top-opportunities`).then(r=>r.json()).then(d=>{
      if(d.setups)setTopSetups(d.setups);if(d.in_play?.stocks)setInPlay(d.in_play.stocks);
      if(d.market_summary)setMktSummary(d.market_summary);if(d.market_open!==undefined)setMktOpen(d.market_open);
    }).catch(()=>{}).finally(()=>setTopLoading(false));
  },[]);

  useEffect(()=>{
    if(!tracked.length)return;
    const f=()=>{fetch(`${API}/api/track-prices?symbols=${tracked.map((x:any)=>x.symbol).join(",")}`).then(r=>r.json()).then(d=>{if(d.prices)setPrices(d.prices)}).catch(()=>{})};
    f();const iv=setInterval(f,300000);return()=>clearInterval(iv);
  },[tracked]);

  const handleScan=useCallback(async()=>{
    setLoading(true);setError("");setScanSetups([]);setChartIdx(null);
    try{const r=await fetch(`${API}/api/scan?symbol=${symbol}&mode=${mode}&ai=true`);const d=await r.json();if(d.error)throw new Error(d.error);setScanSetups(d.setups)}catch(e:any){setError(e.message)}finally{setLoading(false)}
  },[symbol,mode]);

  const scanSymbol=(sym:string)=>{setSymbol(sym);setView("scan");setTimeout(()=>{
    setLoading(true);setError("");setScanSetups([]);setChartIdx(null);
    fetch(`${API}/api/scan?symbol=${sym}&mode=active&ai=true`).then(r=>r.json()).then(d=>{if(!d.error)setScanSetups(d.setups)}).catch(e=>setError(e.message)).finally(()=>setLoading(false));
  },50)};

  const addTrack=(s:any)=>{if(!tracked.find((x:any)=>x.symbol===s.symbol&&x.pattern_name===s.pattern_name))setTracked(p=>[...p,s])};

  const active=view==="opp"?topSetups:scanSetups;
  const filtered=useMemo(()=>{let r=active;if(fBias!=="ALL")r=r.filter((s:any)=>s.bias===fBias.toLowerCase());if(fCat!=="ALL")r=r.filter((s:any)=>s.category===fCat);return[...r].sort((a:any,b:any)=>sortBy==="rr"?b.risk_reward_ratio-a.risk_reward_ratio:(b.composite_score||0)-(a.composite_score||0))},[active,fBias,fCat,sortBy]);

  return(
    <div style={{background:t.bg,minHeight:"100vh",fontFamily:"'Outfit',sans-serif",color:t.text,transition:"background .3s"}}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Outfit:wght@400;500;600;700;800;900&display=swap');*{box-sizing:border-box;margin:0;padding:0}::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:${t.border};border-radius:3px}input:focus,button:focus{outline:none}
      @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
      .fade-in{animation:fadeIn .4s ease-out forwards}
      `}</style>

      {/* Header */}
      <div style={{padding:"10px 24px",borderBottom:`1px solid ${t.border}`,display:"flex",justifyContent:"space-between",alignItems:"center",background:t.bgCard}}>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <BeanLogo size={34}/>
          <span style={{fontSize:22,fontWeight:900,letterSpacing:-.5}}>AlphaBean</span>
          <span style={{fontSize:10,color:t.textMuted,fontWeight:600}}>v3.4</span>
          <div style={{display:"flex",gap:2,marginLeft:14,background:t.border,borderRadius:6,padding:2}}>
            <Pill active={view==="opp"} onClick={()=>setView("opp")} t={t}>Opportunities</Pill>
            <Pill active={view==="scan"} onClick={()=>setView("scan")} t={t}>Scan</Pill>
          </div>
        </div>
        <div style={{display:"flex",gap:10,alignItems:"center",fontSize:12}}>
          <RegimePill r={regime} t={t}/>
          {!mktOpen&&<span style={{fontSize:10,padding:"3px 8px",borderRadius:4,background:t.short+"20",color:t.short,fontWeight:700}}>MARKET CLOSED</span>}
          {active.length>0&&<span style={{color:t.textDim,fontWeight:600}}>{active.filter((s:any)=>s.bias==="long").length}L / {active.filter((s:any)=>s.bias==="short").length}S</span>}
          <button onClick={()=>setBtOpen(true)} style={{fontSize:11,fontWeight:700,padding:"4px 10px",borderRadius:6,border:`1px solid ${t.border}`,background:t.bgCard,color:t.textDim,cursor:"pointer"}}>📊 Backtest</button>
          <button onClick={()=>setDark(!dark)} style={{background:t.border,border:"none",borderRadius:6,padding:"4px 9px",cursor:"pointer",fontSize:13,color:t.text}}>{dark?"☀️":"🌙"}</button>
        </div>
      </div>

      <div style={{padding:"14px 24px"}}>
        {/* Tracked Trades */}
        {tracked.length>0&&<div className="fade-in" style={{background:t.bgCard,border:`1px solid ${t.accent}30`,borderRadius:12,marginBottom:14,overflow:"hidden"}}>
          <div style={{padding:"8px 14px",borderBottom:`1px solid ${t.border}`,fontSize:12,fontWeight:700,color:t.accent,display:"flex",justifyContent:"space-between"}}>
            <span>📡 LIVE TRACKING ({tracked.length})</span><span style={{color:t.textMuted,fontWeight:500,fontSize:11}}>Auto-updates every 5 min</span>
          </div>
          {tracked.map((tr:any,i:number)=><TrackedTrade key={i} trade={tr} price={prices[tr.symbol]} t={t} onRemove={()=>setTracked(p=>p.filter((_:any,idx:number)=>idx!==i))}/>)}
        </div>}

        {/* ═══ OPPORTUNITIES ═══ */}
        {view==="opp"&&<>
          {mktSummary&&<div className="fade-in" style={{padding:"10px 16px",background:t.bgCard,borderRadius:10,border:`1px solid ${t.border}`,marginBottom:12,fontSize:14,color:t.textDim}}><span style={{fontWeight:700,color:t.text}}>Market: </span>{mktSummary}</div>}
          {inPlay.length>0&&<div className="fade-in" style={{marginBottom:14}}>
            <div style={{fontSize:12,fontWeight:700,color:t.textMuted,marginBottom:6,letterSpacing:.4}}>TRENDING ON YAHOO FINANCE</div>
            <div style={{display:"flex",gap:8,overflowX:"auto",paddingBottom:6}}>{inPlay.map((s:any)=><InPlayCard key={s.symbol} stock={s} onClick={()=>scanSymbol(s.symbol)} t={t}/>)}</div>
          </div>}
          {hotStrats.length>0&&<div className="fade-in" style={{display:"flex",gap:6,marginBottom:12,flexWrap:"wrap",alignItems:"center"}}>
            <span style={{fontSize:11,fontWeight:700,color:t.textMuted}}>🔥 HOT STRATEGIES:</span>
            {hotStrats.map((s:any)=><span key={s.name} style={{fontSize:11,padding:"3px 9px",borderRadius:5,background:t.bgCard,border:`1px solid ${t.border}`,color:t.textDim}}>
              <span style={{fontWeight:700,color:t.text}}>{s.name}</span>
              <span style={{marginLeft:4,color:t.long}}>{(s.win_rate*100).toFixed(0)}% WR</span>
              <span style={{marginLeft:4,color:t.accent}}>PF {s.profit_factor?.toFixed(1)}</span>
            </span>)}
          </div>}
          {topLoading&&<LoadingBean t={t} msg="Scanning trending tickers for opportunities..."/>}
        </>}

        {/* ═══ SCAN ═══ */}
        {view==="scan"&&<div style={{display:"flex",gap:10,alignItems:"end",marginBottom:14,flexWrap:"wrap"}}>
          <div><label style={{fontSize:10,color:t.textMuted,display:"block",marginBottom:3,fontWeight:700,letterSpacing:.4}}>SYMBOL</label>
            <input type="text" value={symbol} onChange={e=>setSymbol(e.target.value.toUpperCase())} onKeyDown={e=>e.key==="Enter"&&handleScan()} placeholder="AAPL" style={{fontSize:16,padding:"8px 12px",borderRadius:8,width:110,border:`1.5px solid ${t.border}`,fontWeight:800,background:t.bgCard,color:t.text,fontFamily:"'Outfit',sans-serif"}}/>
          </div>
          <div style={{display:"flex",gap:2,background:t.border,borderRadius:6,padding:2}}><Pill active={mode==="today"} onClick={()=>setMode("today")} t={t} s>Today</Pill><Pill active={mode==="active"} onClick={()=>setMode("active")} t={t} s>Active</Pill></div>
          <button onClick={handleScan} disabled={loading||!symbol} style={{fontSize:14,fontWeight:700,padding:"9px 22px",borderRadius:8,border:"none",background:loading?t.border:t.accent,color:"#fff",cursor:loading?"wait":"pointer",fontFamily:"'Outfit',sans-serif"}}>{loading?"Scanning...":"Scan"}</button>
          <span style={{fontSize:12,color:t.textMuted}}>5m+15m • {pc} patterns • AI</span>
        </div>}

        {error&&<div className="fade-in" style={{padding:"10px 14px",borderRadius:8,background:t.shortBg,color:t.short,fontSize:13,marginBottom:12,border:`1px solid ${t.short}30`}}>{error}</div>}
        {loading&&view==="scan"&&<LoadingBean t={t} msg={`Scanning ${symbol}...`}/>}

        {/* ═══ RESULTS ═══ */}
        {!loading&&active.length>0&&!(view==="opp"&&topLoading)&&<div className="fade-in">
          <div style={{display:"flex",gap:6,marginBottom:10,flexWrap:"wrap",alignItems:"center"}}>
            <div style={{display:"flex",gap:2,background:t.border,borderRadius:6,padding:2}}>{["ALL","LONG","SHORT"].map(b=><Pill key={b} active={fBias===b} onClick={()=>setFBias(b)} t={t} s>{b}</Pill>)}</div>
            <div style={{display:"flex",gap:2,background:t.border,borderRadius:6,padding:2}}>{[["ALL","All"],["classical","Classical"],["candlestick","Candle"],["smb_scalp","SMB"],["quant","Quant"]].map(([v,l])=><Pill key={v} active={fCat===v} onClick={()=>setFCat(v)} t={t} s>{l}</Pill>)}</div>
            <div style={{marginLeft:"auto",display:"flex",gap:2,background:t.border,borderRadius:6,padding:2}}>{[["score","Score"],["rr","R:R"]].map(([k,l])=><Pill key={k} active={sortBy===k} onClick={()=>setSortBy(k as any)} t={t} s>{l}</Pill>)}</div>
          </div>
          <div style={{display:"flex",alignItems:"center",gap:8,padding:"6px 14px",borderBottom:`2px solid ${t.borderLight}`,fontSize:10,fontWeight:700,color:t.textMuted,letterSpacing:.6,textTransform:"uppercase"}}>
            <div style={{width:65}}>Ticker</div><div style={{flex:1}}>Setup / Trigger / Analysis</div>
            <div style={{display:"flex",gap:10}}><div style={{width:55,textAlign:"right"}}>Entry</div><div style={{width:55,textAlign:"right"}}>Stop</div><div style={{width:55,textAlign:"right"}}>Target</div></div>
            <div style={{width:42,textAlign:"center"}}>Score</div><div style={{width:14}}/><div style={{width:70}}/>
          </div>
          {filtered.map((s:any,i:number)=><SetupRow key={`${s.symbol}-${s.pattern_name}-${i}`} s={s} open={chartIdx===i} toggle={()=>setChartIdx(chartIdx===i?null:i)} t={t} onTrack={addTrack}/>)}
          {filtered.length===0&&<div style={{textAlign:"center",padding:30,color:t.textDim,fontSize:14}}>No setups match filters.</div>}
        </div>}

        {!loading&&view==="scan"&&scanSetups.length===0&&!error&&<div style={{textAlign:"center",padding:50,color:t.textDim}}>
          <BeanLogo size={50}/><div style={{fontSize:18,fontWeight:700,marginTop:14,color:t.text}}>Scan a ticker</div>
          <div style={{fontSize:13,marginTop:6}}>{pc} patterns • 6-factor scoring • AI evaluation</div>
        </div>}

        <div style={{textAlign:"center",marginTop:28,padding:"12px 0",borderTop:`1px solid ${t.border}`}}>
          <span style={{fontSize:11,color:t.textMuted}}>AlphaBean v3.4 — {pc} Detectors — Ollama AI — Yahoo Trending — SPY Correlation — All Local, All Free</span>
        </div>
      </div>
      <BacktestModal open={btOpen} onClose={()=>setBtOpen(false)} t={t}/>
    </div>
  );
}