import { useState } from 'react';
import { Send, Activity, AlertCircle, Cpu, Leaf, Flag, Sliders, LayoutGrid, Award, List, Zap } from 'lucide-react';
import './index.css';

interface ApiResponse {
  message?: string;
  prompt: string;
  recompenses?: any;
  modele_recommande?: string;
  score_topsis?: number;
  classement_complet?: [string, number][];
}

const AHP_PROFILES = {
  precision: [
    [1.0, 9.0, 9.0],
    [0.11, 1.0, 1.0],
    [0.11, 1.0, 1.0]
  ],
  green: [
    [1.0, 0.11, 1.0],
    [9.0, 1.0, 9.0],
    [1.0, 0.11, 1.0]
  ],
  sovereignty: [
    [1.0, 1.0, 0.11],
    [1.0, 1.0, 0.11],
    [9.0, 9.0, 1.0]
  ]
};

function App() {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // États pour le Routeur TOPSIS
  const [routingMode, setRoutingMode] = useState<'classic' | 'topsis'>('topsis');
  const [topsisInputMode, setTopsisInputMode] = useState<'cards' | 'sliders'>('cards');
  const [selectedProfile, setSelectedProfile] = useState<'precision' | 'green' | 'sovereignty'>('precision');
  // Les sliders vont de 1 à 10
  const [sliderValues, setSliderValues] = useState({ semantic: 5, eco: 5, sovereignty: 5 });

  const calculateAHPMatrix = (sem: number, eco: number, sov: number) => {
    return [
      [1.0, sem / eco, sem / sov],
      [eco / sem, 1.0, eco / sov],
      [sov / sem, sov / eco, 1.0]
    ];
  };

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>, key: keyof typeof sliderValues) => {
    setSliderValues({ ...sliderValues, [key]: parseInt(e.target.value) });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const apiBaseUrl = import.meta.env.DEV ? 'http://localhost:8000' : '';

      let endpoint = '';
      let body: any = { prompt, limit: 1000 };

      if (routingMode === 'classic') {
        endpoint = '/api/evaluer_prompt';
      } else {
        endpoint = '/api/meilleur_modele';
        body.matrice_ahp = topsisInputMode === 'cards'
          ? AHP_PROFILES[selectedProfile]
          : calculateAHPMatrix(sliderValues.semantic, sliderValues.eco, sliderValues.sovereignty);
      }

      const response = await fetch(`${apiBaseUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        let errorMsg = `Erreur HTTP: ${response.status} ${response.statusText}`;
        try {
          const errData = await response.json();
          if (errData && errData.detail) {
            errorMsg = errData.detail;
          }
        } catch (e) {}
        throw new Error(errorMsg);
      }

      const data = await response.json();
      if (data.message && (!data.recompenses || Object.keys(data.recompenses).length === 0) && !data.modele_recommande) {
        setError(data.message);
        setResult(null);
      } else {
        setResult(data);
      }
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Une erreur est survenue lors de la communication avec l\'API.');
    } finally {
      setLoading(false);
    }
  };


  return (
    <div className="app-container">
      <div className="title-container">
        <h1 className="main-title">AI Engine</h1>
        <p className="subtitle">L'IA pour évaluer et optimiser le routage sémantique de vos prompts</p>
      </div>

      <div className="glass-panel">

        {/* Toggle Mode: Classique vs TOPSIS */}
        <div className="mode-tabs">
          <button
            type="button"
            className={`tab-btn ${routingMode === 'topsis' ? 'active' : ''}`}
            onClick={() => { setRoutingMode('topsis'); setResult(null); }}
          >
            <Zap size={18} /> Routeur TOPSIS
          </button>
          <button
            type="button"
            className={`tab-btn ${routingMode === 'classic' ? 'active' : ''}`}
            onClick={() => { setRoutingMode('classic'); setResult(null); }}
          >
            <Activity size={18} /> Analyse Spécifique
          </button>
        </div>

        <form onSubmit={handleSubmit} className="input-group">

          {/* TOPSIS Configuration Panel */}
          {routingMode === 'topsis' && (
            <div className="topsis-config-panel">
              <div className="sub-mode-tabs">
                <button
                  type="button"
                  className={`sub-tab-btn ${topsisInputMode === 'cards' ? 'active' : ''}`}
                  onClick={() => setTopsisInputMode('cards')}
                >
                  <LayoutGrid size={16} /> Profils Rapides
                </button>
                <button
                  type="button"
                  className={`sub-tab-btn ${topsisInputMode === 'sliders' ? 'active' : ''}`}
                  onClick={() => setTopsisInputMode('sliders')}
                >
                  <Sliders size={16} /> Job Crafting
                </button>
              </div>

              {topsisInputMode === 'cards' ? (
                <div className="profile-cards">
                  <div
                    className={`profile-card ${selectedProfile === 'precision' ? 'active' : ''}`}
                    onClick={() => setSelectedProfile('precision')}
                  >
                    <Cpu size={24} className="profile-icon icon-blue" />
                    <h3>Précision Max</h3>
                    <p>La meilleure réponse possible, sans compromis.</p>
                  </div>
                  <div
                    className={`profile-card green ${selectedProfile === 'green' ? 'active' : ''}`}
                    onClick={() => setSelectedProfile('green')}
                  >
                    <Leaf size={24} className="profile-icon icon-green" />
                    <h3>Éco. & Green IT</h3>
                    <p>Faible empreinte carbone, priorité aux petits modèles.</p>
                  </div>
                  <div
                    className={`profile-card red ${selectedProfile === 'sovereignty' ? 'active' : ''}`}
                    onClick={() => setSelectedProfile('sovereignty')}
                  >
                    <Flag size={24} className="profile-icon icon-red" />
                    <h3>Souveraineté</h3>
                    <p>Privilégie l'Europe et les acteurs français.</p>
                  </div>
                </div>
              ) : (
                <div className="sliders-container">
                  <div className="slider-group">
                    <div className="slider-header">
                      <span className="slider-label" style={{ color: '#38bdf8' }}><Cpu size={16} /> Compétence Sémantique</span>
                      <span className="slider-value">{sliderValues.semantic}/10</span>
                    </div>
                    <input
                      type="range" min="1" max="10"
                      value={sliderValues.semantic}
                      onChange={(e) => handleSliderChange(e, 'semantic')}
                      className="slider-input blue-slider"
                    />
                  </div>
                  <div className="slider-group">
                    <div className="slider-header">
                      <span className="slider-label" style={{ color: '#10b981' }}><Leaf size={16} /> Économie d'Énergie</span>
                      <span className="slider-value">{sliderValues.eco}/10</span>
                    </div>
                    <input
                      type="range" min="1" max="10"
                      value={sliderValues.eco}
                      onChange={(e) => handleSliderChange(e, 'eco')}
                      className="slider-input green-slider"
                    />
                  </div>
                  <div className="slider-group">
                    <div className="slider-header">
                      <span className="slider-label" style={{ color: '#ef4444' }}><Flag size={16} /> Souveraineté Politique</span>
                      <span className="slider-value">{sliderValues.sovereignty}/10</span>
                    </div>
                    <input
                      type="range" min="1" max="10"
                      value={sliderValues.sovereignty}
                      onChange={(e) => handleSliderChange(e, 'sovereignty')}
                      className="slider-input red-slider"
                    />
                  </div>
                </div>
              )}
            </div>
          )}

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
              {loading
                ? (routingMode === 'topsis' ? 'Calcul TOPSIS...' : 'Analyse en cours...')
                : (routingMode === 'topsis' ? 'Trouver le Meilleur Modèle' : 'Évaluer Sémantiquement')}
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
            <div className="loader-text" style={{ marginLeft: '20px', color: 'var(--text-secondary)', fontWeight: 600 }}>
              {routingMode === 'topsis' ? 'Calcul du routage TOPSIS...' : 'Analyse sémantique en cours...'}
            </div>
          </div>
        )}

        {/* --- RÉSULTATS : ANALYSE CLASSIQUE --- */}
        {routingMode === 'classic' && result && result.recompenses && Object.keys(result.recompenses).length > 0 && !loading && (
          <div className="results-container">
            <div className="results-header">
              <Activity size={24} color="#38bdf8" />
              <h2>Classement de l'Analyse Sémantique</h2>
            </div>

            <div className="ranking-section">
              <div className="ranking-list">
                {Object.entries(result.recompenses)
                  // Application de la fonction de tri (décroissant) sur le scalaire score_semantique
                  .sort((a, b) => {
                    const scoreA = (a[1] as any).score_semantique || 0;
                    const scoreB = (b[1] as any).score_semantique || 0;
                    return scoreB - scoreA;
                  })
                  .map(([modele, metriques], index) => {
                    const score = (metriques as any).score_semantique;
                    const volume = (metriques as any).volume_support;

                    return (
                      <div key={modele} className={`ranking-item ${index === 0 ? 'top-1' : ''}`}>
                        <div className="rank">#{index + 1}</div>
                        <div className="model-name" style={{ display: 'flex', flexDirection: 'column' }}>
                          <span>{modele}</span>
                          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 'normal', marginTop: '4px' }}>
                            Volume de support : {volume} évaluation{volume > 1 ? 's' : ''}
                          </span>
                        </div>
                        <div className={`model-score ${score > 0.7 ? 'score-high' : score < 0.3 ? 'score-low' : 'score-medium'}`}>
                          {typeof score === 'number' ? score.toFixed(4) : score}
                        </div>
                      </div>
                    );
                  })}
              </div>
            </div>
          </div>
        )}

        {/* --- RÉSULTATS : MODE TOPSIS --- */}
        {routingMode === 'topsis' && result && result.modele_recommande && !loading && (
          <div className="results-container">
            <div className="winner-card">
              <div className="winner-icon-wrapper">
                <Award size={48} className="winner-icon" />
              </div>
              <div className="winner-info">
                <h3 className="winner-title">Le choix optimal pour votre profil</h3>
                <div className="winner-model">{result.modele_recommande}</div>
                <div className="winner-score">Score TOPSIS de conformité : <strong>{result.score_topsis?.toFixed(4)}</strong></div>
              </div>
            </div>

            {result.classement_complet && (
              <div className="ranking-section" style={{ marginTop: '25px' }}>
                <h3 className="metric-title" style={{ marginBottom: '15px' }}><List size={16} /> Tableau de Classement Complet</h3>
                <div className="ranking-list">
                  {result.classement_complet.map((item, index) => (
                    <div key={item[0]} className={`ranking-item ${index === 0 ? 'top-1' : ''}`}>
                      <div className="rank">#{index + 1}</div>
                      <div className="model-name">{item[0]}</div>
                      <div className="model-score score-high">{item[1].toFixed(4)}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  );
}

export default App;
