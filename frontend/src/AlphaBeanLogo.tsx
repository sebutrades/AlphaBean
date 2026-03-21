import React from "react";
const AlphaBeanLogo: React.FC<{ size?: number }> = ({ size = 200 }) => {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 300 300"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* Background Circle */}
      <circle cx="150" cy="150" r="140" fill="
#F5F5F5" />
      {/* Stock Bars */}
      <rect x="70" y="160" width="20" height="40" fill="
#4CAF50" />
      <rect x="100" y="140" width="20" height="60" fill="
#4CAF50" />
      <rect x="130" y="120" width="20" height="80" fill="
#4CAF50" />
      {/* Arrow */}
      <path
        d="M120 110 L180 70 L175 90 L200 90 L200 110 L175 110 L180 130 Z"
        fill="
#2E7D32"
      />
      {/* Bean Body */}
      <ellipse cx="150" cy="150" rx="55" ry="75" fill="
#8B5A2B" />
      {/* Bean Highlight */}
      <ellipse cx="135" cy="120" rx="15" ry="25" fill="
#A97445" />
      {/* Eyes */}
      <circle cx="135" cy="145" r="8" fill="#000" />
      <circle cx="165" cy="145" r="8" fill="#000" />
      <circle cx="138" cy="142" r="2" fill="#fff" />
      <circle cx="168" cy="142" r="2" fill="#fff" />
      {/* Smile */}
      <path
        d="M135 165 Q150 180 165 165"
        stroke="#000"
        strokeWidth="3"
        fill="none"
        strokeLinecap="round"
      />
      {/* Arms */}
      <path
        d="M100 150 Q80 140 85 170"
        stroke="
#8B5A2B"
        strokeWidth="6"
        fill="none"
        strokeLinecap="round"
      />
      <path
        d="M200 150 Q220 130 210 170"
        stroke="
#8B5A2B"
        strokeWidth="6"
        fill="none"
        strokeLinecap="round"
      />
      {/* Text */}
      <text
        x="150"
        y="260"
        textAnchor="middle"
        fontSize="28"
        fontWeight="bold"
        fill="
#2E7D32"
        fontFamily="Arial, sans-serif"
      >
        Alpha Bean
      </text>
    </svg>
  );
};
export default AlphaBeanLogo;