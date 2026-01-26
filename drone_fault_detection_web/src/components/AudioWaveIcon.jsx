import React from 'react';

const AudioWaveIcon = ({ className = "w-12 h-12" }) => (
  <svg 
    viewBox="0 0 100 60" 
    className={className}
    fill="none" 
    xmlns="http://www.w3.org/2000/svg"
  >
    <rect x="5" y="20" width="6" height="20" rx="2" fill="currentColor" opacity="0.8"/>
    <rect x="15" y="10" width="6" height="40" rx="2" fill="currentColor"/>
    <rect x="25" y="15" width="6" height="30" rx="2" fill="currentColor" opacity="0.9"/>
    <rect x="35" y="5" width="6" height="50" rx="2" fill="currentColor"/>
    <rect x="45" y="12" width="6" height="36" rx="2" fill="currentColor" opacity="0.85"/>
    <rect x="55" y="8" width="6" height="44" rx="2" fill="currentColor"/>
    <rect x="65" y="18" width="6" height="24" rx="2" fill="currentColor" opacity="0.9"/>
    <rect x="75" y="22" width="6" height="16" rx="2" fill="currentColor" opacity="0.7"/>
    <rect x="85" y="25" width="6" height="10" rx="2" fill="currentColor" opacity="0.6"/>
  </svg>
);

export default AudioWaveIcon;
