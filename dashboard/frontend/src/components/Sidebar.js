import React from 'react';

const Sidebar = ({ activeTab, setActiveTab }) => {
  const menuItems = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'alerts', label: 'Alert History', icon: '🔔' },
    { id: 'vision', label: 'Vision Monitor', icon: '👁️' },
    { id: 'drift', label: 'Drift Analysis', icon: '📈' },
  ];

  return (
    <div style={styles.sidebar}>
      <div style={styles.logo}>VAE SYSTEM</div>
      <nav style={styles.nav}>
        {menuItems.map((item) => (
          <button
            key={item.id}
            onClick={() => setActiveTab(item.id)}
            style={{
              ...styles.navItem,
              backgroundColor: activeTab === item.id ? '#2c2e36' : 'transparent',
              color: activeTab === item.id ? '#f7992a' : '#d8d9da',
            }}
          >
            <span style={{ marginRight: '12px' }}>{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>
      <div style={styles.footer}>v1.0.0 | Day 1</div>
    </div>
  );
};

const styles = {
  sidebar: { width: '240px', backgroundColor: '#181b1f', borderRight: '1px solid #2c2e36', display: 'flex', flexDirection: 'column', height: '100vh' },
  logo: { padding: '25px 20px', fontSize: '1.2rem', fontWeight: 'bold', color: '#f7992a', borderBottom: '1px solid #2c2e36' },
  nav: { flexGrow: 1, padding: '15px 0' },
  navItem: { width: '100%', padding: '15px 20px', border: 'none', textAlign: 'left', cursor: 'pointer', fontSize: '0.9rem', display: 'flex', alignItems: 'center' },
  footer: { padding: '15px 20px', fontSize: '0.75rem', color: '#9fa7b3', borderTop: '1px solid #2c2e36' }
};

export default Sidebar;