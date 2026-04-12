import { useState } from 'react';
import { Send, Activity, AlertCircle, Maximize, Target } from 'lucide-react';
import './index.css';

interface ApiResponse {
  message: string;
  prompt: string;
  recompenses: any;
}

function App() {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Pour le mode développement en local, on cible localhost:8000
      // En production avec Nginx, le proxy passera de /api à http://backend:8000/
      const apiBaseUrl = import.meta.env.DEV ? 'http://localhost:8000' : '';
      
      const response = await fetch(`${apiBaseUrl}/api/evaluer_prompt`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: prompt,
          limit: 1000
        }),
      });

      if (!response.ok) {
        throw new Error(`Erreur HTTP: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Une erreur est survenue lors de la communication avec l\'API.');
    } finally {
      setLoading(false);
    }
  };

  const renderRecompenseValue = (value: any) => {
    if (typeof value === 'number') {
      const formattedValue = value.toFixed(4);
      let scoreClass = 'score-medium';
      if (value > 0.7) scoreClass = 'score-high';
      else if (value < 0.3) scoreClass = 'score-low';

      return <span className={`metric-value ${scoreClass}`}>{formattedValue}</span>;
    }
    return <span className="metric-value">{String(value)}</span>;
  }

  return (
    <div className="app-container">
      <div className="title-container">
        <h1 className="main-title">Analyseur Sémantique</h1>
        <p className="subtitle">L'IA pour évaluer la performance de vos prompts selon des récompenses modélisées</p>
      </div>

      <div className="glass-panel">
        <form onSubmit={handleSubmit} className="input-group">
          <div className="textarea-container">
            <textarea
              placeholder="Que voulez-vous exprimer ? Saisissez le prompt à analyser ici..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={loading}
              spellCheck="false"
            />
            <button 
              type="submit" 
              className="submit-btn" 
              disabled={loading || !prompt.trim()}
            >
              {loading ? 'Analyse en cours...' : 'Analyser avec le Routeur AI'}
              <Send size={16} />
            </button>
          </div>
        </form>

        {error && (
          <div className="error-message">
            <AlertCircle size={20} />
            <span>{error}</span>
          </div>
        )}

        {loading && (
          <div className="loader-container">
            <div className="pulse-bubble"></div>
            <div className="pulse-bubble"></div>
            <div className="pulse-bubble"></div>
          </div>
        )}

        {result && result.recompenses && Object.keys(result.recompenses).length > 0 && !loading && (
          <div className="results-container">
            <div className="results-header">
              <Activity size={24} color="#38bdf8" />
              <h2>Indicateurs & Récompenses Estimés</h2>
            </div>
            
            <div className="metrics-grid">
              {Object.entries(result.recompenses).map(([key, value]) => {
                if (typeof value === 'object' && value !== null) return null;
                return (
                  <div key={key} className="metric-card">
                    <div className="metric-title">
                      <Target size={16} />
                      {key.replace(/_/g, ' ')}
                    </div>
                    {renderRecompenseValue(value)}
                  </div>
                );
              })}
            </div>

            {Object.values(result.recompenses).some(v => typeof v === 'object' && v !== null) && (
              <div style={{marginTop: '2rem'}}>
                <h3 className="metric-title" style={{marginBottom: '1rem'}}>
                  <Maximize size={16} /> Structure complète des métriques
                </h3>
                <pre className="json-view">
                  {JSON.stringify(result.recompenses, null, 2)}
                </pre>
              </div>
            )}
            
          </div>
        )}
        
        {result && (!result.recompenses || Object.keys(result.recompenses).length === 0) && !loading && (
          <div className="results-container">
            <div className="error-message" style={{backgroundColor: 'rgba(245, 158, 11, 0.1)', borderColor: '#f59e0b', color: '#f59e0b'}}>
              <AlertCircle size={20} />
              <span>{result.message || "Aucune information sémantique ou récompense trouvée pour ce prompt."}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
