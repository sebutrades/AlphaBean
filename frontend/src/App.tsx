import { useState, useMemo, useEffect, useRef, useCallback } from "react";

const API = "http://localhost:8000";

// ── Theme ──────────────────────────────────────────────────
const DARK = {
  bg:"#080b12",bgCard:"#0f1520",bgHover:"#141c2b",bgModal:"#0c1019",
  border:"#1a2332",borderLight:"#253045",
  text:"#e2e8f0",textDim:"#64748b",textMuted:"#475569",
  accent:"#3b82f6",long:"#10b981",longBg:"rgba(16,185,129,0.12)",
  short:"#ef4444",shortBg:"rgba(239,68,68,0.12)",
  gold:"#f59e0b",goldBg:"rgba(245,158,11,0.12)",
  purple:"#a855f7",
  chartBg:"#080b12",chartGrid:"#111827",chartText:"#64748b",
};
const LIGHT = {
  bg:"#f5f7fa",bgCard:"#ffffff",bgHover:"#f1f5f9",bgModal:"#f0f2f5",
  border:"#e2e8f0",borderLight:"#cbd5e1",
  text:"#1e293b",textDim:"#64748b",textMuted:"#94a3b8",
  accent:"#3b82f6",long:"#16a34a",longBg:"rgba(22,163,74,0.08)",
  short:"#dc2626",shortBg:"rgba(220,38,38,0.08)",
  gold:"#d97706",goldBg:"rgba(217,119,6,0.08)",
  purple:"#7c3aed",
  chartBg:"#ffffff",chartGrid:"#f1f5f9",chartText:"#64748b",
};
type T = typeof DARK;

// ── Bean Logo ──────────────────────────────────────────────
function BeanLogo({size=32}:{size?:number}){return(
  <svg width={size} height={size} viewBox="0 0 100 100" fill="none">
    <ellipse cx="50" cy="54" rx="36" ry="40" fill="#8B5E3C"/>
    <ellipse cx="50" cy="52" rx="33" ry="37" fill="#A0714F"/>
    <ellipse cx="50" cy="50" rx="28" ry="32" fill="#B8865A" opacity="0.3"/>
    <path d="M50 20C50 20 41 42 41 54C41 66 50 88 50 88" stroke="#6B4226" strokeWidth="2.5" strokeLinecap="round" opacity="0.4"/>
    <ellipse cx="38" cy="43" rx="4.5" ry="5" fill="#2D1B0E"/><ellipse cx="62" cy="43" rx="4.5" ry="5" fill="#2D1B0E"/>
    <circle cx="36" cy="41" r="1.8" fill="white"/><circle cx="60" cy="41" r="1.8" fill="white"/>
    <ellipse cx="30" cy="52" rx="5" ry="3" fill="#E8A0A0" opacity="0.35"/>
    <ellipse cx="70" cy="52" rx="5" ry="3" fill="#E8A0A0" opacity="0.35"/>
    <path d="M42 57Q50 66 58 57" stroke="#2D1B0E" strokeWidth="2.5" strokeLinecap="round" fill="none"/>
    <path d="M17 48C14 44 12 38 16 36" stroke="#A0714F" strokeWidth="3" strokeLinecap="round"/>
    <path d="M83 48C86 44 88 38 84 36" stroke="#A0714F" strokeWidth="3" strokeLinecap="round"/>
    <path d="M36 16L32 8" stroke="#10b981" strokeWidth="2.5" strokeLinecap="round"/><circle cx="32" cy="7" r="2.5" fill="#10b981"/>
    <path d="M64 16L68 8" stroke="#ef4444" strokeWidth="2.5" strokeLinecap="round"/><circle cx="68" cy="7" r="2.5" fill="#ef4444"/>
    <path d="M41 64L45 60L49 63L53 58L57 61" stroke="#3b82f6" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.6"/>
  </svg>
)}

// ── Tiny Components ────────────────────────────────────────
function Pill({active,children,onClick,t,s}:{active:boolean;children:React.ReactNode;onClick:()=>void;t:T;s?:boolean}){
  return <button onClick={onClick} style={{fontSize:s?9:10,fontWeight:active?700:500,padding:s?"3px 7px":"4px 10px",borderRadius:5,cursor:"pointer",border:"none",background:active?t.accent:"transparent",color:active?"#fff":t.textDim,fontFamily:"'Outfit',sans-serif",transition:"all .15s"}}>{children}</button>;
}
function Badge({bg,color,children}:{bg:string;color:string;children:React.ReactNode}){
  return <span style={{fontSize:8,fontWeight:700,padding:"1px 5px",borderRadius:3,background:bg,color,textTransform:"uppercase",letterSpacing:.3}}>{children}</span>;
}
const CAT:{[k:string]:{c:string;l:string}}={classical:{c:"#3b82f6",l:"Classical"},candlestick:{c:"#f59e0b",l:"Candle"},smb_scalp:{c:"#a855f7",l:"SMB"},quant:{c:"#10b981",l:"Quant"}};

function CatBadge({cat}:{cat:string}){const c=CAT[cat]||{c:"#888",l:cat};return <Badge bg={c.c+"18"} color={c.c}>{c.l}</Badge>;}
function TfBadge({tf,t}:{tf:string;t:T}){const m=tf.includes("&");return <Badge bg={m?t.accent+"20":t.textDim+"15"} color={m?t.accent:t.textDim}>{m?"★ ":""}{tf}</Badge>;}
function VerdictBadge({v,t}:{v:any;t:T}){if(!v||v.verdict==="PENDING")return null;const cfg:{[k:string]:{bg:string;c:string;i:string}}={CONFIRMED:{bg:t.long+"20",c:t.long,i:"✓"},CAUTION:{bg:t.gold+"20",c:t.gold,i:"!"},DENIED:{bg:t.short+"20",c:t.short,i:"✗"}};const c=cfg[v.verdict]||cfg.CAUTION;return <span title={v.reasoning} style={{fontSize:9,fontWeight:800,padding:"2px 7px",borderRadius:4,background:c.bg,color:c.c,border:`1px solid ${c.c}30`,cursor:"help"}}>{c.i} {v.verdict}</span>;}
function ScoreCell({score,t}:{score:number;t:T}){const c=score>=65?t.long:score>=45?t.gold:t.short;return <span style={{fontFamily:"'JetBrains Mono',monospace",fontSize:12,fontWeight:800,color:c,padding:"2px 5px",borderRadius:4,background:c+"15",display:"inline-block",minWidth:34,textAlign:"center"}}>{score.toFixed(0)}</span>;}
function CorrBadge({corr,t}:{corr:any;t:T}){if(!corr||!corr.grade)return null;const gc:{[k:string]:string}={A:t.long,B:t.accent,C:t.gold,D:t.short,F:t.short};const c=gc[corr.grade]||t.textDim;return <span title={`SPY Correlation: r=${corr.pearson_r}, ${(corr.direction_agreement*100).toFixed(0)}% agreement`} style={{fontSize:8,fontWeight:700,padding:"1px 5px",borderRadius:3,background:c+"18",color:c,cursor:"help"}}>SPY:{corr.grade}</span>;}

const REG:{[k:string]:{l:string;i:string;ck:keyof T}}={trending_bull:{l:"BULL",i:"▲",ck:"long"},trending_bear:{l:"BEAR",i:"▼",ck:"short"},high_volatility:{l:"HI-VOL",i:"◆",ck:"gold"},mean_reverting:{l:"RANGE",i:"◎",ck:"accent"}};
function RegimePill({r,t}:{r:any;t:T}){if(!r?.regime)return null;const c=REG[r.regime];if(!c)return null;const co=t[c.ck] as string;return <span style={{fontSize:9,fontWeight:700,padding:"2px 8px",borderRadius:5,background:co+"15",color:co,border:`1px solid ${co}30`}}>{c.i} {c.l}</span>;}

// ── Backtest Viewer Modal ──────────────────────────────────
function BacktestModal({open,onClose,t}:{open:boolean;onClose:()=>void;t:T}){
  const [data,setData]=useState<any>(null);const [sort,setSort]=useState("edge_score");const [loading,setLoading]=useState(true);
  useEffect(()=>{if(!open)return;setLoading(true);fetch(`${API}/api/backtest/patterns?sort=${sort}`).then(r=>r.json()).then(d=>{setData(d);setLoading(false);}).catch(()=>setLoading(false));},[open,sort]);
  if(!open)return null;
  const summary=data?.summary||{};const patterns=data?.patterns||[];
  return(
    <div style={{position:"fixed",inset:0,zIndex:1000,display:"flex",alignItems:"center",justifyContent:"center",background:"rgba(0,0,0,0.7)",backdropFilter:"blur(4px)"}} onClick={onClose}>
      <div onClick={e=>e.stopPropagation()} style={{background:t.bgModal,border:`1px solid ${t.border}`,borderRadius:14,width:"92vw",maxWidth:1200,maxHeight:"85vh",overflow:"hidden",display:"flex",flexDirection:"column"}}>
        {/* Header */}
        <div style={{padding:"14px 20px",borderBottom:`1px solid ${t.border}`,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
          <div>
            <span style={{fontSize:16,fontWeight:800,color:t.text}}>Backtest Results</span>
            {summary.total_signals&&<span style={{fontSize:11,color:t.textDim,marginLeft:10}}>{summary.total_symbols} symbols • {summary.total_signals} signals • {summary.overall_win_rate}% WR</span>}
          </div>
          <div style={{display:"flex",gap:6,alignItems:"center"}}>
            <span style={{fontSize:9,color:t.textMuted}}>Sort:</span>
            {["edge_score","win_rate","profit_factor","expectancy","total_signals"].map(s=>(
              <Pill key={s} active={sort===s} onClick={()=>setSort(s)} t={t} s>{s.replace("_"," ").replace("edge score","Edge").replace("win rate","WR").replace("profit factor","PF").replace("total signals","Signals")}</Pill>
            ))}
            <button onClick={onClose} style={{background:"none",border:"none",color:t.textDim,fontSize:18,cursor:"pointer",marginLeft:8}}>✕</button>
          </div>
        </div>
        {/* Table */}
        <div style={{overflow:"auto",flex:1,padding:"0 4px"}}>
          {loading?<div style={{padding:40,textAlign:"center",color:t.textDim}}>Loading...</div>:(
            <table style={{width:"100%",borderCollapse:"collapse",fontSize:11,fontFamily:"'JetBrains Mono',monospace"}}>
              <thead><tr style={{position:"sticky",top:0,background:t.bgModal,borderBottom:`2px solid ${t.borderLight}`,fontSize:9,fontWeight:700,color:t.textMuted,textTransform:"uppercase",letterSpacing:.5}}>
                {["Pattern","Cat","Signals","Win%","PF","Exp/R","Avg Win","Avg Loss","Best","Worst","Edge","Grade"].map(h=><th key={h} style={{padding:"8px 6px",textAlign:h==="Pattern"?"left":"right"}}>{h}</th>)}
              </tr></thead>
              <tbody>{patterns.map((p:any)=>{
                const gc:{[k:string]:string}={A:t.long,B:t.accent,C:t.gold,D:t.short,F:t.short};
                const grade=p.edge_score>=70&&p.total_signals>=20?"A":p.edge_score>=55&&p.total_signals>=10?"B":p.edge_score>=40&&p.total_signals>=5?"C":p.edge_score>=25?"D":"F";
                const gc2=gc[grade]||t.textDim;
                return(
                  <tr key={p.name} style={{borderBottom:`1px solid ${t.border}`}}>
                    <td style={{padding:"7px 6px",fontWeight:700,color:t.text,fontFamily:"'Outfit',sans-serif"}}>{p.name}</td>
                    <td style={{textAlign:"right",color:t.textDim}}>{(CAT[PATTERN_CAT[p.name]]||{l:"?"}).l}</td>
                    <td style={{textAlign:"right"}}>{p.total_signals}</td>
                    <td style={{textAlign:"right",color:p.win_rate>=55?t.long:p.win_rate>=45?t.gold:t.short,fontWeight:700}}>{p.win_rate.toFixed(1)}%</td>
                    <td style={{textAlign:"right",color:p.profit_factor>=2?t.long:p.profit_factor>=1?t.text:t.short}}>{p.profit_factor.toFixed(1)}</td>
                    <td style={{textAlign:"right",color:p.expectancy>=0?t.long:t.short}}>{p.expectancy>=0?"+":""}{p.expectancy.toFixed(3)}</td>
                    <td style={{textAlign:"right",color:t.long}}>{p.avg_win_r.toFixed(1)}R</td>
                    <td style={{textAlign:"right",color:t.short}}>{p.avg_loss_r.toFixed(1)}R</td>
                    <td style={{textAlign:"right",color:t.long}}>{p.best_r.toFixed(1)}</td>
                    <td style={{textAlign:"right",color:t.short}}>{p.worst_r.toFixed(1)}</td>
                    <td style={{textAlign:"right"}}><span style={{padding:"1px 5px",borderRadius:3,background:gc2+"18",color:gc2,fontWeight:800}}>{p.edge_score.toFixed(0)}</span></td>
                    <td style={{textAlign:"right"}}><span style={{fontWeight:800,color:gc2}}>{grade}</span></td>
                  </tr>
                );
              })}</tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
// Quick category lookup for backtest table
const PATTERN_CAT:{[k:string]:string}={"Head & Shoulders":"classical","Inverse H&S":"classical","Double Top":"classical","Double Bottom":"classical","Triple Top":"classical","Triple Bottom":"classical","Ascending Triangle":"classical","Descending Triangle":"classical","Symmetrical Triangle":"classical","Bull Flag":"classical","Bear Flag":"classical","Pennant":"classical","Cup & Handle":"classical","Rectangle":"classical","Rising Wedge":"classical","Falling Wedge":"classical","Bullish Engulfing":"candlestick","Bearish Engulfing":"candlestick","Morning Star":"candlestick","Evening Star":"candlestick","Hammer":"candlestick","Shooting Star":"candlestick","Doji":"candlestick","Dragonfly Doji":"candlestick","Three White Soldiers":"candlestick","Three Black Crows":"candlestick","RubberBand Scalp":"smb_scalp","HitchHiker Scalp":"smb_scalp","ORB 15min":"smb_scalp","ORB 30min":"smb_scalp","Second Chance Scalp":"smb_scalp","BackSide Scalp":"smb_scalp","Fashionably Late":"smb_scalp","Spencer Scalp":"smb_scalp","Gap Give & Go":"smb_scalp","Tidal Wave":"smb_scalp","Breaking News":"smb_scalp","Momentum Breakout":"quant","Vol Compression Breakout":"quant","Mean Reversion":"quant","Trend Pullback":"quant","Gap Fade":"quant","Relative Strength Break":"quant","Range Expansion":"quant","Volume Breakout":"quant","VWAP Reversion":"quant","Donchian Breakout":"quant"};

// ── Trade Chart ────────────────────────────────────────────
function TradeChart({setup,onClose,t}:{setup:any;onClose:()=>void;t:T}){
  const ref=useRef<HTMLDivElement>(null);const[loading,setLoading]=useState(true);const[error,setError]=useState("");
  useEffect(()=>{if(!ref.current)return;let chart:any=null;
    (async()=>{try{const lc=await import("lightweight-charts");const tf=setup.timeframe_detected?.includes("15m")?"15min":"5min";const res=await fetch(`${API}/api/chart/${setup.symbol}?timeframe=${tf}&days_back=5`);const data=await res.json();if(data.error)throw new Error(data.error);if(!data.bars?.length)throw new Error("No data");if(ref.current)ref.current.innerHTML="";
      chart=lc.createChart(ref.current!,{width:ref.current!.clientWidth,height:340,layout:{background:{color:t.chartBg}as any,textColor:t.chartText,fontFamily:"JetBrains Mono,monospace"},grid:{vertLines:{color:t.chartGrid},horzLines:{color:t.chartGrid}},crosshair:{mode:lc.CrosshairMode.Normal},rightPriceScale:{borderColor:t.border},timeScale:{borderColor:t.border,timeVisible:true}});
      let s:any;if((lc as any).CandlestickSeries)s=chart.addSeries((lc as any).CandlestickSeries,{upColor:t.long,downColor:t.short,borderUpColor:t.long,borderDownColor:t.short,wickUpColor:t.long,wickDownColor:t.short});else s=chart.addCandlestickSeries({upColor:t.long,downColor:t.short,borderUpColor:t.long,borderDownColor:t.short,wickUpColor:t.long,wickDownColor:t.short});
      s.setData(data.bars);const al=(p:number,c:string,ti:string)=>s.createPriceLine({price:p,color:c,lineWidth:2,lineStyle:lc.LineStyle.Dashed,axisLabelVisible:true,title:ti});al(setup.entry_price,t.accent,"ENTRY");al(setup.stop_loss,t.short,"STOP");al(setup.target_price,t.long,"TARGET");
      Object.entries(setup.key_levels||{}).forEach(([n,p])=>{if(typeof p==="number"&&p>0)s.createPriceLine({price:p,color:t.textMuted,lineWidth:1,lineStyle:lc.LineStyle.Dotted,axisLabelVisible:false,title:n});});
      chart.timeScale().fitContent();setLoading(false);const ro=new ResizeObserver(()=>{if(ref.current&&chart)chart.applyOptions({width:ref.current.clientWidth})});ro.observe(ref.current!);
    }catch(e:any){setError(e.message||"Chart error");setLoading(false);}})();return()=>{if(chart)chart.remove();};},[setup,t]);
  return(<div style={{background:t.bg,border:`1px solid ${t.border}`,borderRadius:10,padding:12,margin:"6px 0"}}>
    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:6}}>
      <span style={{fontSize:12,fontWeight:700,color:t.text}}>{setup.symbol} — {setup.pattern_name}</span>
      <div style={{display:"flex",gap:10,fontSize:10,fontWeight:700,alignItems:"center"}}>
        <span style={{color:t.accent}}>Entry ${setup.entry_price?.toFixed(2)}</span>
        <span style={{color:t.short}}>Stop ${setup.stop_loss?.toFixed(2)}</span>
        <span style={{color:t.long}}>Target ${setup.target_price?.toFixed(2)}</span>
        <button onClick={onClose} style={{background:"none",border:"none",cursor:"pointer",color:t.textDim,fontSize:14}}>✕</button>
      </div>
    </div>
    {loading&&<div style={{textAlign:"center",padding:24,color:t.textDim}}>Loading chart...</div>}
    {error&&<div style={{textAlign:"center",padding:12,color:t.short,fontSize:10}}>Chart: {error}</div>}
    <div ref={ref} style={{width:"100%",minHeight:loading?0:340,marginTop:6}}/>
  </div>);
}

// ── Setup Row ──────────────────────────────────────────────
function SetupRow({s,open,toggle,t,onTrack}:{s:any;open:boolean;toggle:()=>void;t:T;onTrack:(s:any)=>void}){
  const isL=s.bias==="long";const det=new Date(s.detected_at);const isToday=new Date().toDateString()===det.toDateString();
  const timeStr=det.toLocaleTimeString([],{hour:"2-digit",minute:"2-digit"});const dateStr=isToday?"Today":det.toLocaleDateString([],{month:"short",day:"numeric"});
  const trigger=isL?`Buy above $${s.entry_price?.toFixed(2)}`:`Short below $${s.entry_price?.toFixed(2)}`;
  return(<div>
    <div style={{display:"flex",alignItems:"center",gap:6,padding:"8px 10px",borderBottom:`1px solid ${t.border}`,cursor:"pointer",background:open?t.bgHover:"transparent",transition:"background .15s"}}>
      <div onClick={toggle} style={{display:"flex",alignItems:"center",gap:6,flex:1,minWidth:0}}>
        <div style={{width:54}}>
          <div style={{fontSize:13,fontWeight:800,color:t.text}}>{s.symbol}</div>
          <span style={{fontSize:8,fontWeight:700,padding:"1px 5px",borderRadius:3,background:isL?t.longBg:t.shortBg,color:isL?t.long:t.short}}>{s.bias?.toUpperCase()}</span>
        </div>
        <div style={{flex:1,minWidth:100}}>
          <div style={{display:"flex",alignItems:"center",gap:4,flexWrap:"wrap"}}>
            <span style={{fontSize:11,fontWeight:700,color:t.text}}>{s.pattern_name}</span>
            <CatBadge cat={s.category}/><TfBadge tf={s.timeframe_detected} t={t}/><VerdictBadge v={s.ai_verdict} t={t}/>
            <CorrBadge corr={s.spy_correlation} t={t}/>
          </div>
          <div style={{fontSize:9,color:t.textDim,display:"flex",gap:8,marginTop:1}}>
            <span>{dateStr} {timeStr}</span>
            <span style={{color:t.accent,fontWeight:600}}>{trigger}</span>
          </div>
        </div>
        <div style={{display:"flex",gap:6}}>
          {[{l:"ENTRY",v:s.entry_price,c:t.text},{l:"STOP",v:s.stop_loss,c:t.short},{l:"TGT",v:s.target_price,c:t.long}].map(({l,v,c})=>(
            <div key={l} style={{textAlign:"right",minWidth:44}}>
              <div style={{fontSize:7,color:t.textMuted,fontWeight:600,letterSpacing:.3}}>{l}</div>
              <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:11,fontWeight:700,color:c}}>${v?.toFixed(2)}</div>
            </div>
          ))}
        </div>
        <div style={{width:30,textAlign:"center"}}>
          <div style={{fontSize:7,color:t.textMuted}}>R:R</div>
          <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:12,fontWeight:800,color:s.risk_reward_ratio>=2?t.long:t.text}}>{s.risk_reward_ratio?.toFixed(1)}</div>
        </div>
        <ScoreCell score={s.composite_score||0} t={t}/>
        <div style={{width:12,textAlign:"center",fontSize:10,color:t.textDim}}>{open?"▴":"▾"}</div>
      </div>
      {/* Track button */}
      <button onClick={e=>{e.stopPropagation();onTrack(s);}} title="Track this trade" style={{
        background:t.accent+"20",border:`1px solid ${t.accent}40`,borderRadius:5,padding:"3px 7px",
        cursor:"pointer",fontSize:9,fontWeight:700,color:t.accent,whiteSpace:"nowrap",
      }}>+ Track</button>
    </div>
    {open&&<TradeChart setup={s} onClose={toggle} t={t}/>}
  </div>);
}

// ── Live Tracked Trade ─────────────────────────────────────
function TrackedTrade({trade,price,t,onRemove}:{trade:any;price:any;t:T;onRemove:()=>void}){
  const isL=trade.bias==="long";const entry=trade.entry_price;const stop=trade.stop_loss;const target=trade.target_price;
  const cur=price?.price||0;const risk=Math.abs(entry-stop);
  const pnlR=risk>0?(isL?(cur-entry)/risk:(entry-cur)/risk):0;
  const pnlColor=pnlR>=0?t.long:t.short;
  const hitTarget=isL?cur>=target:cur<=target;const hitStop=isL?cur<=stop:cur>=stop;
  const status=hitTarget?"TARGET HIT":hitStop?"STOPPED OUT":"ACTIVE";
  const statusColor=hitTarget?t.long:hitStop?t.short:t.accent;
  return(
    <div style={{display:"flex",alignItems:"center",gap:8,padding:"6px 10px",borderBottom:`1px solid ${t.border}`,fontSize:11}}>
      <div style={{width:50}}><span style={{fontWeight:800,color:t.text}}>{trade.symbol}</span></div>
      <span style={{fontSize:8,padding:"1px 5px",borderRadius:3,background:isL?t.longBg:t.shortBg,color:isL?t.long:t.short,fontWeight:700}}>{trade.bias?.toUpperCase()}</span>
      <span style={{color:t.textDim,flex:1}}>{trade.pattern_name}</span>
      <div style={{fontFamily:"'JetBrains Mono',monospace",fontWeight:700}}>
        <span style={{color:t.textDim}}>Entry ${entry?.toFixed(2)}</span>
        <span style={{color:t.text,marginLeft:8}}>Now ${cur?.toFixed(2)||"..."}</span>
        <span style={{color:pnlColor,marginLeft:8,fontWeight:800}}>{pnlR>=0?"+":""}{pnlR.toFixed(2)}R</span>
      </div>
      <span style={{fontSize:8,fontWeight:800,padding:"2px 6px",borderRadius:3,background:statusColor+"20",color:statusColor}}>{status}</span>
      <button onClick={onRemove} style={{background:"none",border:"none",color:t.textDim,cursor:"pointer",fontSize:12}}>✕</button>
    </div>
  );
}

// ── In-Play Card ───────────────────────────────────────────
function InPlayCard({stock,onClick,t}:{stock:any;onClick:()=>void;t:T}){
  return(
    <div onClick={onClick} style={{padding:"8px 12px",background:t.bgCard,borderRadius:8,border:`1px solid ${t.border}`,cursor:"pointer",minWidth:140,flex:"0 0 auto",transition:"border-color .15s"}}
      onMouseEnter={e=>(e.currentTarget.style.borderColor=t.accent)} onMouseLeave={e=>(e.currentTarget.style.borderColor=t.border)}>
      <span style={{fontSize:13,fontWeight:800,color:t.text}}>{stock.symbol}</span>
      <div style={{fontSize:9,color:t.textDim,marginTop:2,lineHeight:1.3}}>{stock.reason?.slice(0,60)}</div>
    </div>
  );
}

// ── Main App ───────────────────────────────────────────────
export default function App(){
  const[dark,setDark]=useState(true);const t=dark?DARK:LIGHT;
  const[view,setView]=useState<"opp"|"scan">("opp");
  const[symbol,setSymbol]=useState("AAPL");
  const[scanSetups,setScanSetups]=useState<any[]>([]);
  const[topSetups,setTopSetups]=useState<any[]>([]);
  const[inPlay,setInPlay]=useState<any[]>([]);
  const[mktSummary,setMktSummary]=useState("");
  const[loading,setLoading]=useState(false);const[topLoading,setTopLoading]=useState(true);
  const[error,setError]=useState("");const[chartIdx,setChartIdx]=useState<number|null>(null);
  const[fBias,setFBias]=useState("ALL");const[fCat,setFCat]=useState("ALL");const[sortBy,setSortBy]=useState<"score"|"rr">("score");
  const[mode,setMode]=useState<"today"|"active">("today");
  const[regime,setRegime]=useState<any>(null);const[hotStrats,setHotStrats]=useState<any[]>([]);
  const[pc,setPc]=useState(47);const[mktOpen,setMktOpen]=useState(true);
  const[btOpen,setBtOpen]=useState(false);
  // Live tracker
  const[tracked,setTracked]=useState<any[]>([]);const[prices,setPrices]=useState<any>({});

  // Boot
  useEffect(()=>{
    fetch(`${API}/api/health`).then(r=>r.json()).then(d=>{if(d.patterns)setPc(d.patterns);if(d.market_open!==undefined)setMktOpen(d.market_open);}).catch(()=>{});
    fetch(`${API}/api/regime`).then(r=>r.json()).then(d=>{if(d.regime)setRegime(d);}).catch(()=>{});
    fetch(`${API}/api/hot-strategies?top_n=5`).then(r=>r.json()).then(d=>{if(d.strategies)setHotStrats(d.strategies);}).catch(()=>{});
    setTopLoading(true);
    fetch(`${API}/api/top-opportunities`).then(r=>r.json()).then(d=>{
      if(d.setups)setTopSetups(d.setups);if(d.in_play?.stocks)setInPlay(d.in_play.stocks);
      if(d.market_summary)setMktSummary(d.market_summary);if(d.market_open!==undefined)setMktOpen(d.market_open);
    }).catch(()=>{}).finally(()=>setTopLoading(false));
  },[]);

  // Price polling for tracked trades (every 5 min)
  useEffect(()=>{
    if(tracked.length===0)return;
    const fetchPrices=()=>{
      const syms=tracked.map(t=>t.symbol).join(",");
      fetch(`${API}/api/track-prices?symbols=${syms}`).then(r=>r.json()).then(d=>{if(d.prices)setPrices(d.prices);}).catch(()=>{});
    };
    fetchPrices();
    const iv=setInterval(fetchPrices,300000); // 5 min
    return()=>clearInterval(iv);
  },[tracked]);

  const handleScan=useCallback(async()=>{
    setLoading(true);setError("");setScanSetups([]);setChartIdx(null);
    try{const r=await fetch(`${API}/api/scan?symbol=${symbol}&mode=${mode}&ai=true`);const d=await r.json();if(d.error)throw new Error(d.error);setScanSetups(d.setups);}catch(e:any){setError(e.message);}finally{setLoading(false);}
  },[symbol,mode]);

  const handleInPlayClick=(s:any)=>{setSymbol(s.symbol);setView("scan");setTimeout(()=>{
    setLoading(true);setError("");setScanSetups([]);setChartIdx(null);
    fetch(`${API}/api/scan?symbol=${s.symbol}&mode=active&ai=true`).then(r=>r.json()).then(d=>{if(!d.error)setScanSetups(d.setups);}).catch(e=>setError(e.message)).finally(()=>setLoading(false));
  },50);};

  const addTrack=(s:any)=>{if(!tracked.find((t:any)=>t.symbol===s.symbol&&t.pattern_name===s.pattern_name)){setTracked(p=>[...p,s]);}};
  const removeTrack=(i:number)=>setTracked(p=>p.filter((_,idx)=>idx!==i));

  const active=view==="opp"?topSetups:scanSetups;
  const filtered=useMemo(()=>{let r=active;if(fBias!=="ALL")r=r.filter((s:any)=>s.bias===fBias.toLowerCase());if(fCat!=="ALL")r=r.filter((s:any)=>s.category===fCat);return[...r].sort((a:any,b:any)=>sortBy==="rr"?b.risk_reward_ratio-a.risk_reward_ratio:(b.composite_score||0)-(a.composite_score||0));},[active,fBias,fCat,sortBy]);

  const longs=active.filter((s:any)=>s.bias==="long").length;const shorts=active.filter((s:any)=>s.bias==="short").length;

  return(
    <div style={{background:t.bg,minHeight:"100vh",fontFamily:"'Outfit',sans-serif",color:t.text,transition:"background .3s"}}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Outfit:wght@400;500;600;700;800;900&display=swap');*{box-sizing:border-box;margin:0;padding:0}::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:${t.border};border-radius:3px}input:focus,button:focus{outline:none}`}</style>

      {/* Header — full width */}
      <div style={{padding:"8px 20px",borderBottom:`1px solid ${t.border}`,display:"flex",justifyContent:"space-between",alignItems:"center",background:t.bgCard}}>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          <BeanLogo size={28}/>
          <span style={{fontSize:17,fontWeight:900,letterSpacing:-.5}}>AlphaBean</span>
          <span style={{fontSize:8,color:t.textMuted}}>v3.3</span>
          <div style={{display:"flex",gap:2,marginLeft:10,background:t.border,borderRadius:5,padding:2}}>
            <Pill active={view==="opp"} onClick={()=>setView("opp")} t={t}>Opportunities</Pill>
            <Pill active={view==="scan"} onClick={()=>setView("scan")} t={t}>Scan</Pill>
          </div>
        </div>
        <div style={{display:"flex",gap:8,alignItems:"center",fontSize:10}}>
          <RegimePill r={regime} t={t}/>
          {!mktOpen&&<span style={{fontSize:8,padding:"2px 6px",borderRadius:3,background:t.short+"20",color:t.short,fontWeight:700}}>MKT CLOSED</span>}
          {active.length>0&&<><span style={{color:t.long,fontWeight:700}}>{longs}L</span><span style={{color:t.short,fontWeight:700}}>{shorts}S</span></>}
          <button onClick={()=>setBtOpen(true)} style={{fontSize:9,fontWeight:700,padding:"3px 8px",borderRadius:5,border:`1px solid ${t.border}`,background:t.bgCard,color:t.textDim,cursor:"pointer"}}>📊 Backtest</button>
          <button onClick={()=>setDark(!dark)} style={{background:t.border,border:"none",borderRadius:5,padding:"3px 7px",cursor:"pointer",fontSize:11,color:t.text}}>{dark?"☀️":"🌙"}</button>
        </div>
      </div>

      {/* Main — full width with padding */}
      <div style={{padding:"12px 20px"}}>

        {/* Live Tracker Panel */}
        {tracked.length>0&&(
          <div style={{background:t.bgCard,border:`1px solid ${t.border}`,borderRadius:10,marginBottom:12,overflow:"hidden"}}>
            <div style={{padding:"6px 10px",borderBottom:`1px solid ${t.border}`,fontSize:10,fontWeight:700,color:t.accent,display:"flex",justifyContent:"space-between"}}>
              <span>📡 LIVE TRACKING ({tracked.length})</span>
              <span style={{color:t.textMuted,fontWeight:500}}>Updates every 5 min</span>
            </div>
            {tracked.map((tr:any,i:number)=><TrackedTrade key={i} trade={tr} price={prices[tr.symbol]} t={t} onRemove={()=>removeTrack(i)}/>)}
          </div>
        )}

        {/* ═══ OPPORTUNITIES ═══ */}
        {view==="opp"&&<>
          {mktSummary&&<div style={{padding:"8px 12px",background:t.bgCard,borderRadius:8,border:`1px solid ${t.border}`,marginBottom:10,fontSize:11,color:t.textDim}}><span style={{fontWeight:700,color:t.text}}>Market: </span>{mktSummary}</div>}
          {inPlay.length>0&&<div style={{marginBottom:12}}>
            <div style={{fontSize:9,fontWeight:700,color:t.textMuted,marginBottom:5,letterSpacing:.4}}>TRENDING TODAY</div>
            <div style={{display:"flex",gap:6,overflowX:"auto",paddingBottom:4}}>{inPlay.map((s:any)=><InPlayCard key={s.symbol} stock={s} onClick={()=>handleInPlayClick(s)} t={t}/>)}</div>
          </div>}
          {hotStrats.length>0&&<div style={{display:"flex",gap:5,marginBottom:10,flexWrap:"wrap",alignItems:"center"}}>
            <span style={{fontSize:9,fontWeight:700,color:t.textMuted}}>🔥 HOT:</span>
            {hotStrats.map((s:any)=><span key={s.name} style={{fontSize:9,padding:"2px 7px",borderRadius:4,background:t.bgCard,border:`1px solid ${t.border}`,color:t.textDim}}><span style={{fontWeight:700,color:t.text}}>{s.name}</span><span style={{marginLeft:3,color:t.long}}>{(s.win_rate*100).toFixed(0)}%</span></span>)}
          </div>}
          {topLoading&&<div style={{textAlign:"center",padding:50,color:t.textDim}}><BeanLogo size={40}/><div style={{fontSize:12,fontWeight:700,marginTop:12}}>Finding opportunities...</div></div>}
        </>}

        {/* ═══ SCAN ═══ */}
        {view==="scan"&&<div style={{display:"flex",gap:8,alignItems:"end",marginBottom:12,flexWrap:"wrap"}}>
          <div><label style={{fontSize:8,color:t.textMuted,display:"block",marginBottom:2,fontWeight:700,letterSpacing:.4}}>SYMBOL</label>
            <input type="text" value={symbol} onChange={e=>setSymbol(e.target.value.toUpperCase())} onKeyDown={e=>e.key==="Enter"&&handleScan()} placeholder="AAPL" style={{fontSize:13,padding:"6px 10px",borderRadius:6,width:90,border:`1.5px solid ${t.border}`,fontWeight:800,background:t.bgCard,color:t.text,fontFamily:"'Outfit',sans-serif"}}/>
          </div>
          <div style={{display:"flex",gap:2,background:t.border,borderRadius:5,padding:2}}><Pill active={mode==="today"} onClick={()=>setMode("today")} t={t} s>Today</Pill><Pill active={mode==="active"} onClick={()=>setMode("active")} t={t} s>Active</Pill></div>
          <button onClick={handleScan} disabled={loading||!symbol} style={{fontSize:11,fontWeight:700,padding:"7px 18px",borderRadius:6,border:"none",background:loading?t.border:t.accent,color:"#fff",cursor:loading?"wait":"pointer",fontFamily:"'Outfit',sans-serif"}}>{loading?"Scanning...":"Scan"}</button>
          <span style={{fontSize:9,color:t.textMuted}}>5m+15m • {pc} patterns • AI</span>
        </div>}

        {error&&<div style={{padding:"8px 12px",borderRadius:6,background:t.shortBg,color:t.short,fontSize:11,marginBottom:10,border:`1px solid ${t.short}30`}}>{error}</div>}
        {loading&&view==="scan"&&<div style={{textAlign:"center",padding:40,color:t.textDim}}><BeanLogo size={36}/><div style={{fontSize:12,fontWeight:700,marginTop:10}}>Scanning {symbol}...</div></div>}

        {/* ═══ RESULTS ═══ */}
        {!loading&&active.length>0&&!(view==="opp"&&topLoading)&&<>
          <div style={{display:"flex",gap:5,marginBottom:8,flexWrap:"wrap",alignItems:"center"}}>
            <div style={{display:"flex",gap:2,background:t.border,borderRadius:5,padding:2}}>{["ALL","LONG","SHORT"].map(b=><Pill key={b} active={fBias===b} onClick={()=>setFBias(b)} t={t} s>{b}</Pill>)}</div>
            <div style={{display:"flex",gap:2,background:t.border,borderRadius:5,padding:2}}>{[["ALL","All"],["classical","Classical"],["candlestick","Candle"],["smb_scalp","SMB"],["quant","Quant"]].map(([v,l])=><Pill key={v} active={fCat===v} onClick={()=>setFCat(v)} t={t} s>{l}</Pill>)}</div>
            <div style={{marginLeft:"auto",display:"flex",gap:2,background:t.border,borderRadius:5,padding:2}}>{[["score","Score"],["rr","R:R"]].map(([k,l])=><Pill key={k} active={sortBy===k} onClick={()=>setSortBy(k as any)} t={t} s>{l}</Pill>)}</div>
          </div>
          <div style={{display:"flex",alignItems:"center",gap:6,padding:"5px 10px",borderBottom:`1px solid ${t.borderLight}`,fontSize:7,fontWeight:700,color:t.textMuted,letterSpacing:.6,textTransform:"uppercase"}}>
            <div style={{width:54}}>Ticker</div><div style={{flex:1}}>Setup / Trigger / AI / SPY</div>
            <div style={{display:"flex",gap:6}}><div style={{width:44,textAlign:"right"}}>Entry</div><div style={{width:44,textAlign:"right"}}>Stop</div><div style={{width:44,textAlign:"right"}}>Target</div></div>
            <div style={{width:30,textAlign:"center"}}>R:R</div><div style={{width:38,textAlign:"center"}}>Score</div><div style={{width:12}}/><div style={{width:52}}/>
          </div>
          {filtered.map((s:any,i:number)=><SetupRow key={`${s.symbol}-${s.pattern_name}-${i}`} s={s} open={chartIdx===i} toggle={()=>setChartIdx(chartIdx===i?null:i)} t={t} onTrack={addTrack}/>)}
          {filtered.length===0&&<div style={{textAlign:"center",padding:24,color:t.textDim,fontSize:11}}>No setups match filters.</div>}
        </>}

        {!loading&&view==="scan"&&scanSetups.length===0&&!error&&<div style={{textAlign:"center",padding:40,color:t.textDim}}><BeanLogo size={40}/><div style={{fontSize:14,fontWeight:700,marginTop:10,color:t.text}}>Scan a ticker</div><div style={{fontSize:10,marginTop:4}}>{pc} patterns • 6-factor scoring • AI evaluation</div></div>}

        <div style={{textAlign:"center",marginTop:24,padding:"10px 0",borderTop:`1px solid ${t.border}`}}>
          <span style={{fontSize:9,color:t.textMuted}}>AlphaBean v3.3 — {pc} Detectors — Ollama AI — Yahoo Trending — SPY Correlation — All Local</span>
        </div>
      </div>

      <BacktestModal open={btOpen} onClose={()=>setBtOpen(false)} t={t}/>
    </div>
  );
}