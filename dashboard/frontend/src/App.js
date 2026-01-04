// dashboard/frontend/src/App.js
import React, { useState } from 'react';
import './App.css';
import Dashboard from './Dashboard';
import Sidebar from './components/Sidebar';

function App() {
  const [activeTab, setActiveTab] = useState('overview');

  return (
    <div className="app-container">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="main-content">
        <header className="dashboard-header">
          <h1>{activeTab.toUpperCase()} MONITOR</h1>
          <div className="live-indicator">● LIVE STREAMING</div>
        </header>
        {/* Only show the dashboard on the 'overview' tab */}
        {activeTab === 'overview' && <Dashboard />}
        {activeTab !== 'overview' && (
          <div className="panel" style={{textAlign: 'center', padding: '100px'}}>
             <h2>Module Coming in Day 2</h2>
             <p>The {activeTab} analysis logic is scheduled for the next phase.</p>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;