import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';

// 1. MUST REGISTER COMPONENTS OR CHART WILL BE BLANK
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const Dashboard = () => {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        // DIRECT HIT to backend to avoid 404/Proxy issues
        const res = await axios.get('http://localhost:8000/api/history');
        console.log("Data Received:", res.data); // CHECK YOUR BROWSER CONSOLE (F12)
        setHistory([...res.data]);
      } catch (err) {
        console.error("Fetch Error:", err);
      }
    };

    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, []);

  const chartData = {
    labels: history.map((h) => new Date(h.timestamp * 1000).toLocaleTimeString()),
    datasets: [
      {
        label: 'Anomaly Score',
        data: history.map((h) => h.score),
        borderColor: '#5794f2',
        backgroundColor: 'rgba(87, 148, 242, 0.2)',
        fill: true,
        tension: 0.1,
      },
    ],
  };

  return (
    <div style={{ padding: '20px', color: 'white' }}>
      <h2>Real-time Monitor</h2>
      
      {/* CHART SECTION */}
      <div style={{ height: '300px', backgroundColor: '#181b1f', padding: '10px' }}>
        {history.length > 0 ? (
          <Line data={chartData} options={{ maintainAspectRatio: false }} />
        ) : (
          <p>Waiting for data from simulator...</p>
        )}
      </div>

      {/* DEBUG TABLE SECTION */}
      <div style={{ marginTop: '20px' }}>
        <h3>Raw Data (Debug)</h3>
        <table border="1" style={{ width: '100%', borderColor: '#333' }}>
          <thead>
            <tr>
              <th>Time</th>
              <th>Score</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {history.slice(-5).map((h, i) => (
              <tr key={i}>
                <td>{new Date(h.timestamp * 1000).toLocaleTimeString()}</td>
                <td>{h.score.toFixed(4)}</td>
                <td>{h.is_anomaly ? '🔴' : '🟢'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Dashboard;