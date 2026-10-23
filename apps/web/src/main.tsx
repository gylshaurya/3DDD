import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import './app.css';
import { useRoute } from './lib/router';
import { Capture } from './pages/Capture';
import { Home } from './pages/Home';
import { JobPage } from './pages/JobPage';
import { Sandbox } from './pages/Sandbox';
import { ScenePage } from './pages/ScenePage';

function App() {
  const route = useRoute();
  switch (route.page) {
    case 'job':
      return <JobPage key={route.id} id={route.id} />;
    case 'scene':
      return <ScenePage key={route.id} id={route.id} />;
    case 'capture':
      return <Capture />;
    case 'sandbox':
      return <Sandbox />;
    default:
      return <Home />;
  }
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
