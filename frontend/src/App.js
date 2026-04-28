import { useEffect, useState, useRef, useCallback, lazy, Suspense } from 'react';
import '@/App.css';
import { BrowserRouter, Routes, Route, useNavigate, useParams, useLocation } from 'react-router-dom';
import {
  Github,
  Linkedin,
  Mail,
  MapPin,
  Phone,
  ExternalLink,
  Code2,
  Database,
  Brain,
  Cloud,
  Sparkles,
  ChevronDown,
  Send,
  FileText,
  Award,
  Briefcase,
  GraduationCap,
  ArrowLeft,
  X
} from 'lucide-react';

// ═══════════════════════════════════════════════════════════════
// ANTI-GRAVITY PHYSICS ENGINE
// ═══════════════════════════════════════════════════════════════

function useAntiGravity() {
  const mouseRef = useRef({ x: 0, y: 0 });
  const rafRef = useRef(null);
  const elementsRef = useRef([]);

  useEffect(() => {
    const handleMouseMove = (e) => {
      mouseRef.current = { x: e.clientX, y: e.clientY };
    };

    window.addEventListener('mousemove', handleMouseMove, { passive: true });
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const registerElement = useCallback((el, options = {}) => {
    if (el && !elementsRef.current.find(e => e.el === el)) {
      elementsRef.current.push({
        el,
        repelStrength: options.repelStrength || 0.05,
        maxDistance: options.maxDistance || 200,
        currentX: 0,
        currentY: 0,
        targetX: 0,
        targetY: 0,
      });
    }
  }, []);

  useEffect(() => {
    const animate = () => {
      elementsRef.current.forEach((item) => {
        const rect = item.el.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        const dx = mouseRef.current.x - centerX;
        const dy = mouseRef.current.y - centerY;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance < item.maxDistance) {
          const force = (1 - distance / item.maxDistance) * item.repelStrength;
          item.targetX = -dx * force;
          item.targetY = -dy * force;
        } else {
          item.targetX = 0;
          item.targetY = 0;
        }

        // Spring physics interpolation
        item.currentX += (item.targetX - item.currentX) * 0.08;
        item.currentY += (item.targetY - item.currentY) * 0.08;

        if (Math.abs(item.currentX) > 0.01 || Math.abs(item.currentY) > 0.01) {
          item.el.style.transform = `translate(${item.currentX}px, ${item.currentY}px)`;
        }
      });

      rafRef.current = requestAnimationFrame(animate);
    };

    rafRef.current = requestAnimationFrame(animate);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  return { registerElement };
}

// ── Parallax Tilt Hook ────────────────────────────────────────
function useTilt(ref, options = {}) {
  const { maxTilt = 8, scale = 1.02, speed = 400 } = options;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    let animationFrame;

    const handleMouseMove = (e) => {
      const rect = el.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width;
      const y = (e.clientY - rect.top) / rect.height;

      const tiltX = (maxTilt * (0.5 - y)).toFixed(2);
      const tiltY = (maxTilt * (x - 0.5)).toFixed(2);

      cancelAnimationFrame(animationFrame);
      animationFrame = requestAnimationFrame(() => {
        el.style.transform = `perspective(1000px) rotateX(${tiltX}deg) rotateY(${tiltY}deg) scale3d(${scale}, ${scale}, ${scale})`;
        el.style.transition = `transform ${speed}ms cubic-bezier(0.03, 0.98, 0.52, 0.99)`;
      });
    };

    const handleMouseLeave = () => {
      cancelAnimationFrame(animationFrame);
      animationFrame = requestAnimationFrame(() => {
        el.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)';
        el.style.transition = `transform ${speed}ms cubic-bezier(0.03, 0.98, 0.52, 0.99)`;
      });
    };

    el.addEventListener('mousemove', handleMouseMove, { passive: true });
    el.addEventListener('mouseleave', handleMouseLeave, { passive: true });

    return () => {
      el.removeEventListener('mousemove', handleMouseMove);
      el.removeEventListener('mouseleave', handleMouseLeave);
      cancelAnimationFrame(animationFrame);
    };
  }, [ref, maxTilt, scale, speed]);
}

// ── Magnetic Button Hook ──────────────────────────────────────
function useMagnetic(ref, strength = 0.3) {
  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    let animationFrame;

    const handleMouseMove = (e) => {
      const rect = el.getBoundingClientRect();
      const dx = e.clientX - (rect.left + rect.width / 2);
      const dy = e.clientY - (rect.top + rect.height / 2);

      cancelAnimationFrame(animationFrame);
      animationFrame = requestAnimationFrame(() => {
        el.style.transform = `translate(${dx * strength}px, ${dy * strength}px)`;
        el.style.transition = 'transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1)';
      });
    };

    const handleMouseLeave = () => {
      cancelAnimationFrame(animationFrame);
      animationFrame = requestAnimationFrame(() => {
        el.style.transform = 'translate(0px, 0px)';
        el.style.transition = 'transform 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)';
      });
    };

    el.addEventListener('mousemove', handleMouseMove, { passive: true });
    el.addEventListener('mouseleave', handleMouseLeave, { passive: true });

    return () => {
      el.removeEventListener('mousemove', handleMouseMove);
      el.removeEventListener('mouseleave', handleMouseLeave);
      cancelAnimationFrame(animationFrame);
    };
  }, [ref, strength]);
}

// ═══════════════════════════════════════════════════════════════
// PARTICLE NETWORK SYSTEM
// ═══════════════════════════════════════════════════════════════

function ParticleNetwork() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animationFrame;
    let particles = [];

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    const PARTICLE_COUNT = Math.min(60, Math.floor(window.innerWidth / 25));
    const CONNECTION_DISTANCE = 150;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        radius: Math.random() * 1.5 + 0.5,
        opacity: Math.random() * 0.5 + 0.2,
      });
    }

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particles.forEach((p) => {
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(139, 92, 246, ${p.opacity})`;
        ctx.fill();
      });

      // Draw connections
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < CONNECTION_DISTANCE) {
            const opacity = (1 - distance / CONNECTION_DISTANCE) * 0.15;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(139, 92, 246, ${opacity})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }
      }

      animationFrame = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      cancelAnimationFrame(animationFrame);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return <canvas ref={canvasRef} id="particle-canvas" />;
}

// ═══════════════════════════════════════════════════════════════
// TILTABLE CARD COMPONENT
// ═══════════════════════════════════════════════════════════════

function TiltCard({ children, className = '', onClick, style, ...props }) {
  const cardRef = useRef(null);
  useTilt(cardRef, { maxTilt: 6, scale: 1.01 });

  return (
    <div
      ref={cardRef}
      className={`ag-card ${className}`}
      onClick={onClick}
      style={{ ...style, transformStyle: 'preserve-3d' }}
      {...props}
    >
      {children}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// MAGNETIC BUTTON COMPONENT
// ═══════════════════════════════════════════════════════════════

function MagneticButton({ children, className = '', onClick, type, disabled, ...props }) {
  const btnRef = useRef(null);
  useMagnetic(btnRef, 0.25);

  return (
    <button
      ref={btnRef}
      className={className}
      onClick={onClick}
      type={type}
      disabled={disabled}
      {...props}
    >
      {children}
    </button>
  );
}

// ═══════════════════════════════════════════════════════════════
// PROJECT DATA
// ═══════════════════════════════════════════════════════════════

const projectsData = {
  'credit-default': {
    id: 1,
    slug: 'credit-default',
    title: 'Credit Default Risk Analyzer',
    tagline: 'Predicting customer credit repayment default using machine learning',
    year: '2024',
    category: 'Machine Learning',
    description: `Credit default risk modeling is a critical task for financial institutions. This project aims to predict the likelihood of a customer defaulting on their credit card payment in the next month based on their demographic data and 6-month transaction history.\n\nObjective: Maximize the identification of high-risk customers (Recall) while maintaining a reasonable precision to avoid unnecessary declines.`,
    tech: ['Python', 'XGBoost', 'Scikit-learn', 'Streamlit', 'AWS SageMaker', 'Docker', 'Pandas', 'Flask API'],
    github: 'https://github.com/i-aryann/Credit-Default-Prediction',
    demo: 'https://credit-default-prediction-aryan.streamlit.app/',
    highlights: ['92% ROC-AUC Score', 'SHAP Integration', 'Automated CI/CD Pipeline'],
    features: [
      {
        title: 'High Accuracy Ensemble',
        desc: 'Utilized an ensemble of XGBoost and Random Forest models to achieve a 92% ROC-AUC score, significantly outperforming baseline logistic regression.'
      },
      {
        title: 'Explainable AI',
        desc: 'Integrated SHAP (SHapley Additive exPlanations) values to provide transparency, showing exactly why a specific customer was flagged as high-risk.'
      },
      {
        title: 'Business Metrics',
        desc: 'Engineered real world business metrics to enhance performance and business impact, including Utilization Ratio, Delinquency Trend, Payment-to-Balance Ratio and Spending Volatility.'
      },
      {
        title: 'Automated Pipeline',
        desc: 'Implemented a fully automated CI/CD pipeline for model retraining and deployment using AWS SageMaker and GitHub Actions.'
      }
    ]
  },
  'nlp-sentiment': {
    id: 2,
    slug: 'nlp-sentiment',
    title: 'NLP Sentiment Analyzer',
    tagline: 'Understanding Social Media Sentiment with Transformers',
    year: '2023',
    category: 'NLP',
    description: `In the age of social media, understanding public opinion is vital for brand management. This tool analyzes social media feeds to determine the sentiment (positive, negative, neutral) regarding specific topics or brand mentions in real-time.\n\nBuilt upon the BERT architecture, the model understands context and nuance better than traditional bag-of-words approaches. It processes thousands of tweets per minute and visualizes the aggregate sentiment trends over time.`,
    tech: ['NLP', 'BERT', 'Transformers', 'FastAPI', 'Kafka', 'React', 'Hugging Face', 'Python'],
    github: '#',
    demo: '#',
    highlights: ['BERT Architecture', 'Real-time Processing', 'Interactive Dashboard'],
    features: [
      {
        title: 'Transformer Architecture',
        desc: 'Uses a fine-tuned BERT model for state-of-the-art accuracy in understanding context and sarcasm.'
      },
      {
        title: 'Real-time Processing',
        desc: 'Ingests and processes live Twitter/X data streams using Apache Kafka and Spark Streaming.'
      },
      {
        title: 'Interactive Visualization',
        desc: 'Frontend dashboard built with React and D3.js to visualize sentiment shifts and trending keywords.'
      }
    ]
  },
  'sales-forecasting': {
    id: 3,
    slug: 'sales-forecasting',
    title: 'Sales Forecasting Dashboard',
    tagline: 'Predicting Future Revenue with Time Series Analysis',
    year: '2023',
    category: 'Data Analytics',
    description: `Accurate sales forecasting is key to inventory management and resource planning. This project provides a comprehensive dashboard that predicts future sales based on historical data, seasonality, and market trends.\n\nUsing Long Short-Term Memory (LSTM) recurrent neural networks, the model captures complex temporal dependencies. The results are presented in an intuitive dashboard that allows business managers to run 'what-if' scenarios.`,
    tech: ['Time Series', 'LSTM', 'TensorFlow', 'Streamlit', 'PostgreSQL', 'Plotly', 'Keras', 'Pandas'],
    github: '#',
    demo: '#',
    highlights: ['LSTM Networks', 'What-if Scenarios', 'Automated Reporting'],
    features: [
      {
        title: 'LSTM Networks',
        desc: 'Implemented Recurrent Neural Networks (RNN) specifically designed to learn long-term dependencies in time-series data.'
      },
      {
        title: 'Interactive Scenarios',
        desc: 'Users can adjust parameters (e.g., marketing spend, pricing) to see potential impacts on future sales.'
      },
      {
        title: 'Automated Reporting',
        desc: 'Generates weekly PDF reports summarizing forecast accuracy and highlighting significant deviations.'
      }
    ]
  }
};

// ═══════════════════════════════════════════════════════════════
// LOADING COMPONENT
// ═══════════════════════════════════════════════════════════════

function PageLoader() {
  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      background: '#0A0A0F',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 9999
    }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{
          width: 48,
          height: 48,
          border: '3px solid rgba(139, 92, 246, 0.2)',
          borderTop: '3px solid #8B5CF6',
          borderRadius: '50%',
          animation: 'spin 1s linear infinite',
          margin: '0 auto 16px'
        }} />
        <p style={{ color: '#94A3B8', fontSize: '0.9rem' }}>Loading...</p>
        <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// PAGE TRANSITION
// ═══════════════════════════════════════════════════════════════

function PageTransition({ children }) {
  const location = useLocation();
  const [displayLocation, setDisplayLocation] = useState(location);
  const [transitionStage, setTransitionStage] = useState('fadeIn');

  useEffect(() => {
    if (location !== displayLocation) {
      setTransitionStage('fadeOut');
    }
  }, [location, displayLocation]);

  return (
    <div
      className={`page-transition ${transitionStage}`}
      onAnimationEnd={() => {
        if (transitionStage === 'fadeOut') {
          setTransitionStage('fadeIn');
          setDisplayLocation(location);
        }
      }}
    >
      {children}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// ANIMATED BACKGROUND
// ═══════════════════════════════════════════════════════════════

function AnimatedBackground() {
  return (
    <>
      <div className="ag-bg-base" />
      <div className="ag-orb ag-orb-1" />
      <div className="ag-orb ag-orb-2" />
      <div className="ag-orb ag-orb-3" />
      <div className="ag-orb ag-orb-4" />
      <ParticleNetwork />
      <div className="grid-overlay" />
    </>
  );
}

// ═══════════════════════════════════════════════════════════════
// MAIN PORTFOLIO COMPONENT
// ═══════════════════════════════════════════════════════════════

function Portfolio() {
  const [activeSection, setActiveSection] = useState('home');
  const [isScrolled, setIsScrolled] = useState(false);
  const [formStatus, setFormStatus] = useState('');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const navigate = useNavigate();
  const ctaRef = useRef(null);

  useMagnetic(ctaRef, 0.15);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);

      const sections = ['home', 'skills', 'projects', 'experience', 'contact'];
      const current = sections.find(section => {
        const element = document.getElementById(section);
        if (element) {
          const rect = element.getBoundingClientRect();
          return rect.top <= 150 && rect.bottom >= 150;
        }
        return false;
      });
      if (current) setActiveSection(current);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('animate-in');
          }
        });
      },
      { threshold: 0.1 }
    );

    document.querySelectorAll('.scroll-reveal').forEach((el) => observer.observe(el));

    return () => observer.disconnect();
  }, []);

  const handleMobileNavClick = (sectionId) => {
    setMobileMenuOpen(false);
    scrollToSection(sectionId);
  };

  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      const offset = 80;
      const elementPosition = element.getBoundingClientRect().top;
      const offsetPosition = elementPosition + window.pageYOffset - offset;

      window.scrollTo({
        top: offsetPosition,
        behavior: 'smooth'
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setFormStatus('sending');

    setTimeout(() => {
      setFormStatus('success');
      e.target.reset();
      setTimeout(() => setFormStatus(''), 3000);
    }, 1000);
  };

  const skills = [
    {
      category: 'Languages',
      icon: <Code2 className="w-7 h-7" />,
      items: ['Python', 'SQL', 'JavaScript']
    },
    {
      category: 'AI & Machine Learning',
      icon: <Brain className="w-7 h-7" />,
      items: ['TensorFlow', 'PyTorch', 'Scikit-Learn', 'XGBoost', 'Random Forest', 'K-Means']
    },
    {
      category: 'Deep Learning & NLP',
      icon: <Sparkles className="w-7 h-7" />,
      items: ['Transformers', 'BERT', 'LSTMs', 'Hugging Face', 'Keras']
    },
    {
      category: 'MLOps & Cloud',
      icon: <Cloud className="w-7 h-7" />,
      items: ['AWS SageMaker', 'AWS EC2', 'AWS S3', 'AWS Lambda', 'Docker', 'CI/CD Pipelines']
    },
    {
      category: 'Data & Analytics',
      icon: <Database className="w-7 h-7" />,
      items: ['Pandas', 'NumPy', 'Power BI', 'Matplotlib', 'Seaborn', 'Plotly']
    },
    {
      category: 'Databases',
      icon: <Database className="w-7 h-7" />,
      items: ['PostgreSQL', 'MySQL', 'MongoDB']
    }
  ];

  const projects = Object.values(projectsData);

  const experience = [
    {
      period: 'April 2024 - Present',
      role: 'Data Analyst',
      company: 'National Skill Development Corporation',
      location: 'New Delhi',
      icon: <Briefcase className="w-5 h-5" />,
      highlights: [
        'Engineered SQL-backed dashboards to audit 100K+ CRM records, achieving 4.5% YoY performance increase',
        'Automated data pipelines using Python and SQL, reducing manual reporting time by 30%',
        'Developed data-driven optimization heuristics, driving 89% adherence to TAT across teams'
      ]
    },
    {
      period: '2022 - 2023',
      role: 'Data Science Specialization',
      company: 'Scaler',
      location: 'Bangalore',
      icon: <Award className="w-5 h-5" />,
      highlights: []
    },
    {
      period: '2021 - 2022',
      role: 'Research Engineer',
      company: 'VIVO Mobiles India Limited',
      location: 'Noida',
      icon: <Briefcase className="w-5 h-5" />,
      highlights: [
        'Led a 15-member team with KPI-driven dashboards, resulting in 4% productivity increase',
        'Created data visualization dashboards enabling real-time decision making'
      ]
    },
    {
      period: '2018 - 2022',
      role: 'Bachelor of Technology',
      company: 'Dr. A.P.J. Abdul Kalam Technical University',
      location: 'India',
      icon: <GraduationCap className="w-5 h-5" />,
      highlights: ['Specialization in Electronics and Communication Engineering']
    }
  ];

  const navItems = ['home', 'skills', 'projects', 'experience', 'contact'];

  return (
    <div className="App">
      <AnimatedBackground />

      {/* ── Navigation ──────────────────────────────────────── */}
      <nav
        className={`ag-nav ${isScrolled ? 'ag-nav-scrolled' : ''}`}
        data-testid="main-navigation"
      >
        <div style={{
          maxWidth: 1280,
          margin: '0 auto',
          padding: '16px 24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <button
            onClick={() => navigate('/')}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              fontSize: '1.6rem',
              fontWeight: 800,
              letterSpacing: '-0.02em'
            }}
            className="ag-gradient-text"
            data-testid="logo-button"
          >
            ARYAN<span style={{
              WebkitTextFillColor: 'rgba(148, 163, 184, 0.5)',
              fontWeight: 300
            }}>.ai</span>
          </button>

          {/* Desktop Menu */}
          <div className="hidden md:flex" style={{ alignItems: 'center', gap: 32 }}>
            {navItems.map((section) => (
              <button
                key={section}
                onClick={() => { navigate('/'); setTimeout(() => scrollToSection(section), 100); }}
                className={`ag-link ${activeSection === section ? 'ag-link-active' : ''}`}
                style={{
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  textTransform: 'capitalize',
                  fontSize: '0.85rem',
                  fontWeight: 500,
                  padding: '4px 0',
                  fontFamily: 'inherit'
                }}
                data-testid={`nav-${section}`}
              >
                {section}
              </button>
            ))}
            <a
              href="#resume"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 8,
                padding: '8px 20px',
                borderRadius: 10,
                border: '1px solid rgba(139, 92, 246, 0.3)',
                background: 'rgba(139, 92, 246, 0.08)',
                color: '#A78BFA',
                fontSize: '0.85rem',
                fontWeight: 600,
                textDecoration: 'none',
                transition: 'all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)'
              }}
              onMouseEnter={e => {
                e.currentTarget.style.background = 'rgba(139, 92, 246, 0.15)';
                e.currentTarget.style.borderColor = 'rgba(139, 92, 246, 0.5)';
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 4px 20px rgba(139, 92, 246, 0.2)';
              }}
              onMouseLeave={e => {
                e.currentTarget.style.background = 'rgba(139, 92, 246, 0.08)';
                e.currentTarget.style.borderColor = 'rgba(139, 92, 246, 0.3)';
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = 'none';
              }}
              data-testid="resume-button"
            >
              <FileText className="w-4 h-4" />
              Resume
            </a>
          </div>

          {/* Mobile Menu Button */}
          <button
            className="md:hidden"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              color: 'white',
              zIndex: 50,
              padding: 8
            }}
            data-testid="mobile-menu-button"
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? (
              <X className="w-6 h-6" />
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                <div style={{ width: 22, height: 2, background: 'white', borderRadius: 1 }} />
                <div style={{ width: 22, height: 2, background: 'white', borderRadius: 1 }} />
                <div style={{ width: 16, height: 2, background: 'white', borderRadius: 1 }} />
              </div>
            )}
          </button>
        </div>

        {/* Mobile Menu Overlay */}
        {mobileMenuOpen && (
          <div className="md:hidden mobile-menu-slide" style={{
            position: 'fixed',
            inset: 0,
            top: 64,
            background: 'rgba(10, 10, 15, 0.97)',
            backdropFilter: 'blur(30px)',
            zIndex: 40,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 32,
            padding: 24
          }}>
            {navItems.map((section) => (
              <button
                key={section}
                onClick={() => handleMobileNavClick(section)}
                style={{
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  textTransform: 'capitalize',
                  fontSize: '1.8rem',
                  fontWeight: 600,
                  color: activeSection === section ? '#06B6D4' : '#94A3B8',
                  transition: 'color 0.3s ease',
                  fontFamily: 'inherit'
                }}
              >
                {section}
              </button>
            ))}
            <a
              href="#resume"
              onClick={() => setMobileMenuOpen(false)}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 8,
                padding: '12px 28px',
                borderRadius: 12,
                border: '1px solid rgba(139, 92, 246, 0.3)',
                background: 'rgba(139, 92, 246, 0.08)',
                color: '#A78BFA',
                fontSize: '1.1rem',
                fontWeight: 600,
                textDecoration: 'none'
              }}
            >
              <FileText className="w-5 h-5" />
              Resume
            </a>
          </div>
        )}
      </nav>

      {/* ── Hero Section ────────────────────────────────────── */}
      <section id="home" style={{
        position: 'relative',
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '0 24px'
      }} data-testid="hero-section">
        <div style={{ maxWidth: 900, margin: '0 auto', textAlign: 'center', zIndex: 10, position: 'relative' }}>
          <div className="scroll-reveal">
            <h1 className="ag-hero-title">
              <span style={{ color: 'white' }}>Hi, I'm </span>
              <span className="ag-gradient-text">Aryan</span>
            </h1>
            <p className="ag-hero-subtitle">
              I build scalable machine learning and artificial intelligence systems with automated MLOps pipelines and cloud-native deployment strategies.
            </p>
            <div className="ag-cta-glow" style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'center', gap: 16 }}>
              <MagneticButton
                onClick={() => scrollToSection('projects')}
                className="ag-btn-magnetic ag-btn-primary"
                data-testid="view-work-button"
              >
                View My Work
              </MagneticButton>
              <MagneticButton
                onClick={() => scrollToSection('contact')}
                className="ag-btn-magnetic ag-btn-outline"
                data-testid="contact-button"
              >
                Get In Touch
              </MagneticButton>
            </div>
          </div>
        </div>

        <button
          onClick={() => scrollToSection('skills')}
          className="ag-scroll-indicator"
          data-testid="scroll-indicator"
          style={{ background: 'none', border: 'none', cursor: 'pointer' }}
        >
          <ChevronDown className="w-6 h-6" style={{ color: '#06B6D4', animation: 'scrollDots 2s ease-in-out infinite' }} />
        </button>
      </section>

      {/* ── Skills Section ──────────────────────────────────── */}
      <section id="skills" style={{ position: 'relative', padding: '96px 24px', zIndex: 2 }} data-testid="skills-section">
        <div style={{ maxWidth: 1280, margin: '0 auto' }}>
          <div className="scroll-reveal" style={{ textAlign: 'center', marginBottom: 64 }}>
            <p className="ag-section-label">What I Do</p>
            <h2 className="ag-section-title">Skills & Expertise</h2>
            <p className="ag-section-subtitle">
              Specialized in building end-to-end AI solutions from research to production deployment
            </p>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))',
            gap: 24
          }}>
            {skills.map((skill, index) => (
              <TiltCard
                key={index}
                className="scroll-reveal"
                style={{ padding: 28, animationDelay: `${index * 100}ms` }}
                data-testid={`skill-card-${index}`}
              >
                <div style={{ position: 'relative', zIndex: 2 }}>
                  <div style={{ color: '#06B6D4', marginBottom: 16 }}>{skill.icon}</div>
                  <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'white', marginBottom: 16 }}>
                    {skill.category}
                  </h3>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                    {skill.items.map((item, i) => (
                      <span key={i} className="ag-skill-pill">{item}</span>
                    ))}
                  </div>
                </div>
              </TiltCard>
            ))}
          </div>
        </div>
      </section>

      {/* ── Projects Section ────────────────────────────────── */}
      <section id="projects" style={{ position: 'relative', padding: '96px 24px', zIndex: 2 }} data-testid="projects-section">
        <div style={{ maxWidth: 1280, margin: '0 auto' }}>
          <div className="scroll-reveal" style={{ textAlign: 'center', marginBottom: 64 }}>
            <p className="ag-section-label">My Work</p>
            <h2 className="ag-section-title">Featured Projects</h2>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(360px, 1fr))',
            gap: 32
          }}>
            {projects.map((project, index) => (
              <TiltCard
                key={project.id}
                onClick={() => navigate(`/project/${project.slug}`)}
                className="scroll-reveal"
                style={{
                  padding: 32,
                  cursor: 'pointer',
                  animationDelay: `${index * 150}ms`
                }}
                data-testid={`project-card-${project.id}`}
              >
                <div style={{ position: 'relative', zIndex: 2 }}>
                  <div className="ag-project-number">0{project.id}</div>
                  <h3 style={{
                    fontSize: '1.5rem',
                    fontWeight: 700,
                    color: 'white',
                    marginBottom: 12,
                    transition: 'color 0.3s ease'
                  }}>
                    {project.title}
                  </h3>
                  <p style={{ color: '#94A3B8', marginBottom: 16, lineHeight: 1.7 }} className="line-clamp-3">
                    {project.description.split('\n\n')[0]}
                  </p>

                  <div style={{ marginBottom: 16 }}>
                    {project.highlights.slice(0, 2).map((highlight, i) => (
                      <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                        <div className="ag-highlight-dot" />
                        <span style={{ fontSize: '0.85rem', color: '#67E8F9' }}>{highlight}</span>
                      </div>
                    ))}
                  </div>

                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 24 }}>
                    {project.tech.slice(0, 4).map((tech, i) => (
                      <span key={i} className="ag-tech-badge">{tech}</span>
                    ))}
                    {project.tech.length > 4 && (
                      <span className="ag-tech-badge">+{project.tech.length - 4} more</span>
                    )}
                  </div>

                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    fontSize: '0.85rem',
                    fontWeight: 600,
                    color: '#A78BFA'
                  }}>
                    View Details <ExternalLink className="w-4 h-4" />
                  </div>
                </div>
              </TiltCard>
            ))}
          </div>
        </div>
      </section>

      {/* ── Experience Section ──────────────────────────────── */}
      <section id="experience" style={{ position: 'relative', padding: '96px 24px', zIndex: 2 }} data-testid="experience-section">
        <div style={{ maxWidth: 900, margin: '0 auto' }}>
          <div className="scroll-reveal" style={{ textAlign: 'center', marginBottom: 64 }}>
            <p className="ag-section-label">My Journey</p>
            <h2 className="ag-section-title">Experience & Education</h2>
          </div>

          <div style={{ position: 'relative' }}>
            <div className="ag-timeline-line" />

            {experience.map((exp, index) => (
              <div
                key={index}
                className="scroll-reveal"
                style={{
                  position: 'relative',
                  paddingLeft: 48,
                  paddingBottom: index === experience.length - 1 ? 0 : 48,
                  animationDelay: `${index * 100}ms`
                }}
                data-testid={`experience-item-${index}`}
              >
                <div className="ag-timeline-node" style={{ top: 24 }} />

                <TiltCard style={{ padding: 24 }}>
                  <div style={{ position: 'relative', zIndex: 2 }}>
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16, marginBottom: 16 }}>
                      <div style={{ color: '#8B5CF6', marginTop: 4 }}>{exp.icon}</div>
                      <div style={{ flex: 1 }}>
                        <p style={{ color: '#06B6D4', fontSize: '0.85rem', fontWeight: 600, marginBottom: 8 }}>
                          {exp.period}
                        </p>
                        <h3 style={{ fontSize: '1.15rem', fontWeight: 700, color: 'white', marginBottom: 4 }}>
                          {exp.role}
                        </h3>
                        <p style={{ color: '#94A3B8', fontSize: '0.9rem' }}>
                          {exp.company} • {exp.location}
                        </p>
                      </div>
                    </div>
                    {exp.highlights.length > 0 && (
                      <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 8 }}>
                        {exp.highlights.map((highlight, i) => (
                          <li key={i} style={{
                            color: '#94A3B8',
                            fontSize: '0.85rem',
                            lineHeight: 1.7,
                            display: 'flex',
                            alignItems: 'flex-start',
                            gap: 8,
                          }}>
                            <span className="ag-bullet">▸</span>
                            <span>{highlight}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </TiltCard>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Contact Section ─────────────────────────────────── */}
      <section id="contact" style={{ position: 'relative', padding: '96px 24px', zIndex: 2 }} data-testid="contact-section">
        <div style={{ maxWidth: 1100, margin: '0 auto' }}>
          <div className="scroll-reveal" style={{ textAlign: 'center', marginBottom: 64 }}>
            <p className="ag-section-label">Get In Touch</p>
            <h2 className="ag-section-title">Let's Work Together</h2>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 400px), 1fr))',
            gap: 32
          }}>
            {/* Contact Info */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {[
                { icon: <Mail className="w-5 h-5" />, iconClass: 'ag-contact-icon-purple', label: 'Email', value: 'aryangupta.7263@gmail.com', href: 'mailto:aryangupta.7263@gmail.com', testId: 'contact-email' },
                { icon: <Phone className="w-5 h-5" />, iconClass: 'ag-contact-icon-pink', label: 'Phone', value: '+91 7534090544', href: 'tel:+917534090544', testId: 'contact-phone' },
                { icon: <MapPin className="w-5 h-5" />, iconClass: 'ag-contact-icon-cyan', label: 'Location', value: 'New Delhi, India', testId: 'contact-location' },
                { icon: <Linkedin className="w-5 h-5" />, iconClass: 'ag-contact-icon-purple', label: 'LinkedIn', value: 'linkedin.com/in/aryangupta7263', href: 'https://www.linkedin.com/in/aryangupta7263', external: true, testId: 'contact-linkedin' },
              ].map((item, i) => (
                <TiltCard key={i} className="scroll-reveal" style={{ padding: 20 }} data-testid={item.testId}>
                  <div style={{ position: 'relative', zIndex: 2, display: 'flex', alignItems: 'center', gap: 16 }}>
                    <div className={`ag-contact-icon ${item.iconClass}`}>
                      {item.icon}
                    </div>
                    <div>
                      <p style={{ color: '#64748B', fontSize: '0.8rem', marginBottom: 4 }}>{item.label}</p>
                      {item.href ? (
                        <a
                          href={item.href}
                          target={item.external ? '_blank' : undefined}
                          rel={item.external ? 'noopener noreferrer' : undefined}
                          className="ag-link"
                          style={{ color: 'white', fontWeight: 500 }}
                        >
                          {item.value}
                        </a>
                      ) : (
                        <p style={{ color: 'white', fontWeight: 500, margin: 0 }}>{item.value}</p>
                      )}
                    </div>
                  </div>
                </TiltCard>
              ))}
            </div>

            {/* Contact Form */}
            <TiltCard className="scroll-reveal" style={{ padding: 32 }} data-testid="contact-form">
              <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 20, position: 'relative', zIndex: 2 }}>
                <input
                  type="text"
                  name="name"
                  placeholder="Your Name"
                  required
                  className="ag-input"
                  data-testid="form-name"
                />
                <input
                  type="email"
                  name="email"
                  placeholder="Your Email"
                  required
                  className="ag-input"
                  data-testid="form-email"
                />
                <textarea
                  name="message"
                  placeholder="Your Message"
                  rows="5"
                  required
                  className="ag-input"
                  style={{ resize: 'none' }}
                  data-testid="form-message"
                />
                <MagneticButton
                  type="submit"
                  disabled={formStatus === 'sending'}
                  className={`ag-btn-magnetic ag-btn-primary ${formStatus === 'success' ? 'ag-success-glow' : ''}`}
                  style={{ width: '100%', justifyContent: 'center' }}
                  data-testid="form-submit"
                >
                  {formStatus === 'sending' ? (
                    'Sending...'
                  ) : formStatus === 'success' ? (
                    '✓ Message Sent!'
                  ) : (
                    <>
                      <Send className="w-4 h-4" />
                      Send Message
                    </>
                  )}
                </MagneticButton>
              </form>
            </TiltCard>
          </div>
        </div>
      </section>

      {/* ── Footer ──────────────────────────────────────────── */}
      <footer className="ag-footer" style={{ position: 'relative', padding: '32px 24px', zIndex: 2 }} data-testid="footer">
        <div style={{ maxWidth: 1280, margin: '0 auto', textAlign: 'center' }}>
          <p style={{ color: '#475569', fontSize: '0.85rem' }}>
            © 2025 Aryan Gupta. Built with React & Tailwind CSS.
          </p>
        </div>
      </footer>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// PROJECT DETAIL PAGE
// ═══════════════════════════════════════════════════════════════

function ProjectDetail() {
  const { slug } = useParams();
  const navigate = useNavigate();
  const project = projectsData[slug];

  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);

  const handleBackClick = () => {
    navigate('/');
    setTimeout(() => {
      const projectsSection = document.getElementById('projects');
      if (projectsSection) {
        const offset = 80;
        const elementPosition = projectsSection.getBoundingClientRect().top;
        const offsetPosition = elementPosition + window.pageYOffset - offset;
        window.scrollTo({
          top: offsetPosition,
          behavior: 'smooth'
        });
      }
    }, 100);
  };

  if (!project) {
    return (
      <div style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#0A0A0F'
      }}>
        <div style={{ textAlign: 'center' }}>
          <h1 style={{ fontSize: '2.5rem', fontWeight: 800, color: 'white', marginBottom: 16 }}>
            Project Not Found
          </h1>
          <MagneticButton
            onClick={() => navigate('/')}
            className="ag-btn-magnetic ag-btn-primary"
          >
            Go Back Home
          </MagneticButton>
        </div>
      </div>
    );
  }

  return (
    <div className="App" style={{ minHeight: '100vh' }}>
      <AnimatedBackground />

      {/* Navigation */}
      <nav className="ag-nav ag-nav-scrolled">
        <div style={{
          maxWidth: 1280,
          margin: '0 auto',
          padding: '16px 24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <button
            onClick={() => navigate('/')}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              fontSize: '1.6rem',
              fontWeight: 800
            }}
            className="ag-gradient-text"
          >
            ARYAN<span style={{
              WebkitTextFillColor: 'rgba(148, 163, 184, 0.5)',
              fontWeight: 300
            }}>.ai</span>
          </button>
          <button
            onClick={handleBackClick}
            className="ag-link"
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              color: '#A78BFA',
              fontSize: '0.9rem',
              fontWeight: 500,
              fontFamily: 'inherit'
            }}
          >
            <ArrowLeft className="w-5 h-5" />
            Back to Portfolio
          </button>
        </div>
      </nav>

      {/* Project Hero */}
      <section style={{ position: 'relative', paddingTop: 128, paddingBottom: 64, padding: '128px 24px 64px' }}>
        <div style={{ maxWidth: 900, margin: '0 auto', textAlign: 'center', zIndex: 10, position: 'relative' }}>
          <div style={{ marginBottom: 24, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12, flexWrap: 'wrap' }}>
            <span className="ag-tech-badge" style={{ padding: '6px 16px', fontSize: '0.85rem' }}>
              {project.category}
            </span>
            <span className="ag-skill-pill" style={{ background: 'rgba(139, 92, 246, 0.08)' }}>
              {project.year}
            </span>
          </div>
          <h1 className="ag-gradient-text" style={{
            fontSize: 'clamp(2.5rem, 6vw, 4.5rem)',
            fontWeight: 900,
            marginBottom: 24,
            lineHeight: 1.1
          }}>
            {project.title}
          </h1>
          <p style={{ fontSize: 'clamp(1.1rem, 2vw, 1.35rem)', color: '#94A3B8', marginBottom: 32, maxWidth: 700, margin: '0 auto 32px' }}>
            {project.tagline}
          </p>

          <div className="ag-cta-glow" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 16, flexWrap: 'wrap', marginBottom: 16 }}>
            <a
              href={project.github}
              target="_blank"
              rel="noopener noreferrer"
              className="ag-btn-magnetic ag-btn-primary"
              style={{ textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: 10 }}
              data-testid="project-github-button"
            >
              <Github className="w-5 h-5" />
              View Code
            </a>
            <a
              href={project.demo}
              target="_blank"
              rel="noopener noreferrer"
              className="ag-btn-magnetic ag-btn-outline"
              style={{ textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: 10 }}
              data-testid="project-demo-button"
            >
              <ExternalLink className="w-5 h-5" />
              Live Demo
            </a>
          </div>
        </div>
      </section>

      {/* Project Content */}
      <section style={{ position: 'relative', padding: '64px 24px', zIndex: 2 }}>
        <div style={{ maxWidth: 900, margin: '0 auto' }}>
          {/* Overview */}
          <TiltCard style={{ padding: 32, marginBottom: 32 }}>
            <div style={{ position: 'relative', zIndex: 2 }}>
              <h2 style={{
                fontSize: '1.75rem',
                fontWeight: 800,
                color: 'white',
                marginBottom: 24,
                paddingLeft: 16,
                borderLeft: '3px solid #8B5CF6'
              }}>
                Overview
              </h2>
              {project.description.split('\n\n').map((para, i) => (
                <p key={i} style={{ color: '#CBD5E1', fontSize: '1.05rem', lineHeight: 1.8, marginBottom: i < project.description.split('\n\n').length - 1 ? 16 : 0 }}>
                  {para}
                </p>
              ))}
            </div>
          </TiltCard>

          {/* Key Features */}
          <TiltCard style={{ padding: 32, marginBottom: 32 }}>
            <div style={{ position: 'relative', zIndex: 2 }}>
              <h2 style={{
                fontSize: '1.75rem',
                fontWeight: 800,
                color: 'white',
                marginBottom: 24,
                paddingLeft: 16,
                borderLeft: '3px solid #06B6D4'
              }}>
                Key Features
              </h2>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 300px), 1fr))',
                gap: 20
              }}>
                {project.features.map((feature, i) => (
                  <div key={i} className="ag-card" style={{ padding: 24, borderRadius: 12 }}>
                    <div style={{ position: 'relative', zIndex: 2 }}>
                      <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#06B6D4', marginBottom: 12 }}>
                        {feature.title}
                      </h3>
                      <p style={{ color: '#94A3B8', lineHeight: 1.7, fontSize: '0.9rem' }}>
                        {feature.desc}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </TiltCard>

          {/* Tech Stack */}
          <TiltCard style={{ padding: 32 }}>
            <div style={{ position: 'relative', zIndex: 2 }}>
              <h2 style={{
                fontSize: '1.75rem',
                fontWeight: 800,
                color: 'white',
                marginBottom: 24,
                paddingLeft: 16,
                borderLeft: '3px solid #A78BFA'
              }}>
                Tech Stack
              </h2>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
                {project.tech.map((tech, i) => (
                  <span key={i} className="ag-tech-badge" style={{ padding: '8px 18px', fontSize: '0.9rem' }}>
                    {tech}
                  </span>
                ))}
              </div>
            </div>
          </TiltCard>
        </div>
      </section>

      {/* Footer */}
      <footer className="ag-footer" style={{ position: 'relative', padding: '32px 24px', marginTop: 64, zIndex: 2 }}>
        <div style={{ maxWidth: 1280, margin: '0 auto', textAlign: 'center' }}>
          <p style={{ color: '#475569', fontSize: '0.85rem' }}>
            © 2025 Aryan Gupta. Built with React & Tailwind CSS.
          </p>
        </div>
      </footer>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// APP ROOT
// ═══════════════════════════════════════════════════════════════

function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<PageLoader />}>
        <Routes>
          <Route path="/" element={<Portfolio />} />
          <Route path="/project/:slug" element={<ProjectDetail />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}

export default App;
