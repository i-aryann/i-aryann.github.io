import { useEffect, useState, lazy, Suspense, useRef } from 'react';
import './App.css';
import { BrowserRouter, Routes, Route, useNavigate, useParams, useLocation } from 'react-router-dom';
import { Button } from './components/ui/button';
import { Card } from './components/ui/card';
import { Badge } from './components/ui/badge';
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
  X,
  Layers
} from 'lucide-react';

// Project Data with full details
const projectsData = {
  'LLM-RAG': {
    id: 1,
    slug: 'LLM-RAG',
    title: 'Compliance RAG System',
    cardTagline: 'RAG system using LLM API, LangChain and Qdrant DB, served via FastAPI, deployed on AWS with CI/CD deployment.',
    tagline: 'Cutting compliance rules research time for banking institutions with AI-powered retrieval over complex regulations',
    year: '2026',
    category: 'Generative AI',
    description: `A compliance research assistant using a Retrieval-Augmented Generation (RAG) pipeline to extract precise answers from large regulatory documents.\n\n The system combines semantic vector search(Qdrant DB) with keyword-based retrieval (BM25) to enable hybrid search, improving recall across complex queries.\n\n Retrieved results are further refined using re-ranking (cross-encoder models) to prioritize the most relevant context before passing it to LLM via LangChain.\n\n The system includes token-aware chunking, context compression, and Citation-Based responses to ensure accuracy and traceability—critical for compliance workflows.\n\n Built with FastAPI and deployed on Amazon Web Services EC2 using Docker and Nginx, it delivers low-latency, scalable, and production-ready performance.`,
    tech: ['LLM', "RAG", 'Vector DB', 'Hybrid Search', "Pydantic", "LangChain", 'FastAPI', 'AWS Cloud', "Docker", "Github Actions", 'Python'],
    github: 'https://github.com/i-aryann/ComplianceBrain-RAG',
    demo: 'https://compliance-rag.duckdns.org/',
    architecture: '/regulatory_rag.png', // TODO: Add your image name here after uploading to public/ folder (e.g., '/nlp-arch.png')
    highlights: ['Hybrid Retrieval + Re-ranking Pipeline', 'Source-Cited Compliance Answers'],
    features: [
      {
        title: 'Source-Cited Answers',
        desc: 'Provides responses with traceable document references to ensure transparency and compliance reliability.'
      },
      {
        title: 'Token-Optimized Context Handling',
        desc: 'Efficiently selects and compresses context to reduce cost while maintaining answer accuracy.'
      },
      {
        title: 'Hybrid Search Retrieval and Re-ranking',
        desc: ' First combines semantic and keyword search to improve recall and then prioritizes most relevant results using deep relevance scoring before generating final responses'
      },
      {
        title: 'Cloud Deployment & CI/CD Automation',
        desc: 'Ensures scalable deployment with automated build, testing, and seamless continuous delivery updates over AWS Cloud.'
      }
    ]
  },
  'credit-default': {
    id: 2,
    slug: 'credit-default',
    title: 'Credit Default Risk Analyzer',
    cardTagline: 'Predicting customer credit repayment default with ML',
    tagline: 'Predicting customer credit repayment default using machine learning',
    year: '2024',
    category: 'Machine Learning',
    description: `Credit default risk modeling is a critical task for financial institutions. This project aims to predict the likelihood of a customer defaulting on their credit card payment in the next month based on their demographic data and 6-month transaction history.\n\nObjective: Maximize the identification of high-risk customers (Recall) while maintaining a reasonable precision to avoid unnecessary declines.`,
    tech: ['Python', 'XGBoost', 'Scikit-learn', 'Streamlit', 'AWS SageMaker', 'Docker', 'Pandas', 'Flask API'],
    github: 'https://github.com/i-aryann/Credit-Default-Prediction',
    demo: 'https://credit-default-prediction-aryan.streamlit.app/',
    architecture: '', // TODO: Add your image name here after uploading to public/ folder (e.g., '/credit-arch.png')
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

  'sales-forecasting': {
    id: 3,
    slug: 'sales-forecasting',
    title: 'Sales Forecasting Dashboard',
    cardTagline: 'Predicting Future Revenue with Time Series Analysis',
    tagline: 'Predicting Future Revenue with Time Series Analysis',
    year: '2023',
    category: 'Data Analytics',
    description: `Accurate sales forecasting is key to inventory management and resource planning. This project provides a comprehensive dashboard that predicts future sales based on historical data, seasonality, and market trends.\n\nUsing Long Short-Term Memory (LSTM) recurrent neural networks, the model captures complex temporal dependencies. The results are presented in an intuitive dashboard that allows business managers to run 'what-if' scenarios.`,
    tech: ['Time Series', 'LSTM', 'TensorFlow', 'Streamlit', 'PostgreSQL', 'Plotly', 'Keras', 'Pandas'],
    github: '#',
    demo: '#',
    architecture: '', // TODO: Add your image name here after uploading to public/ folder (e.g., '/sales-arch.png')
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

// Loading component
function PageLoader() {
  return (
    <div className="fixed inset-0 bg-black flex items-center justify-center z-50">
      <div className="text-center">
        <div className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
        <p className="text-white text-lg">Loading...</p>
      </div>
    </div>
  );
}

// Page transition wrapper
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

// ─── SEO helper – updates <title>, meta tags & canonical per route ────────────
const updateSEO = ({ title, description, canonical }) => {
  // Title
  document.title = title;

  // Meta description
  const metaDesc = document.querySelector('meta[name="description"]');
  if (metaDesc) metaDesc.setAttribute('content', description);

  // Canonical link
  const url = canonical || 'https://aryangupta.work/';
  let canonicalEl = document.querySelector('link[rel="canonical"]');
  if (canonicalEl) canonicalEl.setAttribute('href', url);

  // Open Graph
  const ogUrl = document.querySelector('meta[property="og:url"]');
  const ogTitle = document.querySelector('meta[property="og:title"]');
  const ogDesc = document.querySelector('meta[property="og:description"]');
  if (ogUrl) ogUrl.setAttribute('content', url);
  if (ogTitle) ogTitle.setAttribute('content', title);
  if (ogDesc) ogDesc.setAttribute('content', description);

  // Twitter
  const twTitle = document.querySelector('meta[name="twitter:title"]');
  const twDesc = document.querySelector('meta[name="twitter:description"]');
  if (twTitle) twTitle.setAttribute('content', title);
  if (twDesc) twDesc.setAttribute('content', description);
};

// ─── Central GA4 tracking helper ────────────────────────────────────────────
const trackEvent = (eventName, params = {}) => {
  if (window.gtag) {
    window.gtag('event', eventName, params);
  }
};

function Portfolio() {
  const [activeSection, setActiveSection] = useState('home');
  const [isScrolled, setIsScrolled] = useState(false);
  const [formStatus, setFormStatus] = useState('');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const navigate = useNavigate();
  const sectionViewedRef = useRef({});

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);

      const sections = ['home', 'projects', 'skills', 'experience', 'contact'];
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

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // ─── Set page SEO on mount ────────────────────────────────────────────────
  useEffect(() => {
    updateSEO({
      title: 'Aryan Gupta | AI Engineer · LLM · RAG · MLOps Portfolio',
      description:
        'Aryan Gupta — AI Engineer specialising in LLMs, Retrieval-Augmented Generation (RAG), MLOps, and end-to-end machine-learning systems. Explore projects in NLP, Deep Learning, AWS SageMaker, and production ML pipelines.',
      canonical: 'https://aryangupta.work/',
    });
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

  // ─── Track section views (scroll depth) ──────────────────────────────────
  useEffect(() => {
    const sections = ['home', 'projects', 'skills', 'experience', 'contact'];
    const sectionObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !sectionViewedRef.current[entry.target.id]) {
            sectionViewedRef.current[entry.target.id] = true;
            trackEvent('section_viewed', {
              section_name: entry.target.id,
              event_category: 'Engagement',
            });
          }
        });
      },
      { threshold: 0.4 }
    );
    sections.forEach((id) => {
      const el = document.getElementById(id);
      if (el) sectionObserver.observe(el);
    });
    return () => sectionObserver.disconnect();
  }, []);

  // ─── Time-on-page engagement milestones ──────────────────────────────────
  useEffect(() => {
    const t30 = setTimeout(() => trackEvent('engaged_30s', { event_category: 'Engagement' }), 30000);
    const t60 = setTimeout(() => trackEvent('engaged_60s', { event_category: 'Engagement' }), 60000);
    const t3m = setTimeout(() => trackEvent('engaged_3min', { event_category: 'Engagement' }), 180000);
    return () => { clearTimeout(t30); clearTimeout(t60); clearTimeout(t3m); };
  }, []);

  // Close mobile menu on section click
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
    trackEvent('contact_form_submitted', {
      event_category: 'Contact',
      event_label: 'Portfolio Contact Form',
    });

    const form = e.target;
    const data = new FormData(form);

    try {
      // ⚠️ Formspree endpoint injected
      const response = await fetch('https://formspree.io/f/xqeynyyb', {
        method: 'POST',
        body: data,
        headers: {
          'Accept': 'application/json'
        }
      });

      if (response.ok) {
        setFormStatus('success');
        form.reset();
        setTimeout(() => setFormStatus(''), 3000);
      } else {
        setFormStatus('error');
        setTimeout(() => setFormStatus(''), 3000);
      }
    } catch (error) {
      setFormStatus('error');
      setTimeout(() => setFormStatus(''), 3000);
    }
  };

  const skills = [
    {
      category: 'Programming',
      icon: <Code2 className="w-8 h-8" />,
      items: ['Python', 'SQL']
    },
    {
      category: 'LLM & AI',
      icon: <Sparkles className="w-8 h-8" />,
      items: ['OpenAI & Open-source APIs', 'LangChain', 'LangGraph', 'AI Agents', 'Hugging Face']
    },
    {
      category: 'Databases & Vector DB',
      icon: <Database className="w-8 h-8" />,
      items: ['Qdrant', 'FAISS', 'PostgreSQL', 'MySQL', 'MongoDB', 'Redis']
    },
    {
      category: 'Frameworks & APIs',
      icon: <Layers className="w-8 h-8" />,
      items: ['FastAPI', 'Flask', 'Pydantic', 'langsmith', 'NumPy', 'Pandas']
    },
    {
      category: 'Cloud & DevOps',
      icon: <Cloud className="w-8 h-8" />,
      items: ['AWS Cloud (EC2, S3, Lambda, SageMaker etc)', 'Docker', 'GitHub Actions', 'CI/CD pipelines']
    },
    {
      category: 'Machine Learning & NLP',
      icon: <Brain className="w-8 h-8" />,
      items: ['Classical ML (supervised and unsupervised)', 'Scikit-learn', 'TensorFlow', 'Transformers', 'BERT']
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

  return (
    <div className="App">
      <main role="main">
      {/* Animated Background */}
      <div className="fixed inset-0 z-0 overflow-hidden">
        <div className="absolute inset-0 bg-black" />
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse-slow" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-pink-500/20 rounded-full blur-3xl animate-pulse-slower" />
        <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse-slowest" />
        <div className="grid-overlay" />
      </div>

      {/* Sticky Navigation */}
      <nav
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${isScrolled ? 'bg-black/80 backdrop-blur-lg border-b border-purple-500/20 shadow-lg shadow-purple-500/5' : 'bg-transparent'
          }`}
        data-testid="main-navigation"
      >
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between relative">
          <button
            onClick={() => navigate('/')}
            className="text-2xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent hover:scale-105 transition-transform"
            data-testid="logo-button"
          >
            ARYAN<span className="text-white/50 font-light">.ai</span>
          </button>

          {/* Desktop Menu Center Links */}
          <div className="hidden lg:flex absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 items-center gap-8">
            {['home', 'projects', 'skills', 'experience'].map((section) => (
              <button
                key={section}
                onClick={() => {
                  trackEvent('nav_click', { event_category: 'Navigation', section_name: section });
                  navigate('/');
                  setTimeout(() => scrollToSection(section), 100);
                }}
                className={`capitalize text-sm font-medium transition-all ${activeSection === section
                  ? 'text-pink-400'
                  : 'text-gray-400 hover:text-white'
                  }`}
                data-testid={`nav-${section}`}
              >
                {section}
              </button>
            ))}
          </div>

          {/* Desktop Menu Right Buttons */}
          <div className="hidden md:flex items-center gap-4 ml-auto">
            <button
              onClick={() => {
                trackEvent('nav_click', { event_category: 'Navigation', section_name: 'contact' });
                navigate('/');
                setTimeout(() => scrollToSection('contact'), 100);
              }}
              className="px-4 py-2 rounded-lg border border-purple-500/50 bg-purple-500/10 text-purple-400 hover:bg-purple-500/20 transition-all text-sm font-medium flex items-center gap-2"
              data-testid="nav-contact"
            >
              Start a Conversation
            </button>
            <a
              href={`/Resume.pdf?t=${Date.now()}`}
              target="_blank"
              rel="noopener noreferrer"
              onClick={() => trackEvent('resume_viewed', { event_category: 'Resume', event_label: 'Desktop Nav' })}
              className="px-4 py-2 rounded-lg border border-purple-500/50 bg-purple-500/10 text-purple-400 hover:bg-purple-500/20 transition-all text-sm font-medium flex items-center justify-center gap-2"
              data-testid="resume-button"
            >
              Get My Resume
            </a>
          </div>

          {/* Mobile Menu Button */}
          <button
            className="md:hidden text-white z-50"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            data-testid="mobile-menu-button"
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? (
              <X className="w-6 h-6" />
            ) : (
              <div className="space-y-1.5">
                <div className="w-6 h-0.5 bg-white" />
                <div className="w-6 h-0.5 bg-white" />
                <div className="w-6 h-0.5 bg-white" />
              </div>
            )}
          </button>
        </div>

        {/* Mobile Menu Overlay */}
        {mobileMenuOpen && (
          <div className="md:hidden fixed inset-0 top-16 bg-black/95 backdrop-blur-lg z-40 mobile-menu-slide">
            <div className="flex flex-col items-center justify-center h-full gap-8 px-6">
              {['home', 'projects', 'skills', 'experience'].map((section) => (
                <button
                  key={section}
                  onClick={() => handleMobileNavClick(section)}
                  className={`capitalize text-2xl font-medium transition-all ${activeSection === section
                    ? 'text-pink-400'
                    : 'text-gray-400 hover:text-white'
                    }`}
                >
                  {section}
                </button>
              ))}
              <button
                onClick={() => handleMobileNavClick('contact')}
                className="px-6 py-3 rounded-lg border border-purple-500/50 bg-purple-500/10 text-purple-400 text-lg font-medium flex items-center gap-2 transition-all justify-center w-full max-w-xs mx-auto"
              >
                Start a Conversation
              </button>
              <a
                href={`/Resume.pdf?t=${Date.now()}`}
                target="_blank"
                rel="noopener noreferrer"
                onClick={() => {
                  setMobileMenuOpen(false);
                  trackEvent('resume_viewed', { event_category: 'Resume', event_label: 'Mobile Nav' });
                }}
                className="px-6 py-3 rounded-lg border border-purple-500/50 bg-purple-500/10 text-purple-400 text-lg font-medium flex items-center gap-2 transition-all justify-center w-full max-w-xs mx-auto"
              >
                Get My Resume
              </a>
            </div>
          </div>
        )}
      </nav>

      {/* Hero Section */}
      <section id="home" className="relative min-h-screen flex items-center justify-center px-6" data-testid="hero-section">
        <div className="max-w-5xl mx-auto text-center z-10">
          <div className="scroll-reveal">
            <h1 className="text-6xl md:text-8xl font-bold mb-8 leading-tight">
              <span className="text-white">Hi, I'm </span>
              <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent animate-gradient">
                Aryan
              </span>
            </h1>
            <p className="text-xl md:text-2xl text-gray-300 mb-12 max-w-3xl mx-auto leading-relaxed">
              <span className="font-bold">
                I architect Generative and Agentic AI systems
              </span>
              <br />
              <span className="text-lg md:text-xl mt-4 inline-block text-gray-400">
                • LLM Fine-Tuning • Advanced RAG • Scalable Cloud MLOps
              </span>
            </p>
            <div className="flex flex-wrap items-center justify-center gap-4">
              <Button
                onClick={() => {
                  trackEvent('cta_click', { event_category: 'Hero', event_label: 'View My Work' });
                  scrollToSection('projects');
                }}
                className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white px-8 py-6 text-lg rounded-lg shadow-lg hover:shadow-purple-500/50 transition-all font-bold tracking-wide"
                data-testid="view-work-button"
              >
                Try Live Demos
              </Button>
              <Button
                onClick={() => {
                  trackEvent('cta_click', { event_category: 'Hero', event_label: 'Get In Touch' });
                  scrollToSection('contact');
                }}
                className="relative overflow-hidden group border-2 border-pink-500/50 bg-pink-500/10 hover:bg-pink-500/30 text-white px-8 py-6 text-lg rounded-lg backdrop-blur-sm transition-all shadow-[0_0_15px_rgba(236,72,153,0.4)] hover:shadow-[0_0_25px_rgba(236,72,153,0.7)] hover:-translate-y-1"
                data-testid="contact-button"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-pink-500/0 via-purple-500/30 to-blue-500/0 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
                <span className="relative z-10 font-bold tracking-wide">Let's Build Together</span>
              </Button>
            </div>
          </div>
        </div>

        <button
          onClick={() => scrollToSection('skills')}
          className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce cursor-pointer z-10"
          data-testid="scroll-indicator"
        >
          <ChevronDown className="w-8 h-8 text-pink-400" />
        </button>
      </section>

      {/* Projects Section */}
      <section id="projects" className="relative py-24 px-6" data-testid="projects-section">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16 scroll-reveal">
            <p className="text-pink-400 text-sm uppercase tracking-widest mb-4 font-semibold">My Work</p>
            <h2 className="text-5xl md:text-6xl font-bold text-white mb-6">Featured Projects</h2>

          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-8" style={{ gridTemplateRows: 'auto' }}>
            {projects.map((project, index) => (
              <Card
                key={project.id}
                onClick={() => {
                  trackEvent('project_card_clicked', {
                    event_category: 'Projects',
                    project_name: project.title,
                    project_slug: project.slug,
                  });
                  navigate(`/project/${project.slug}`);
                }}
                className="scroll-reveal bg-gradient-to-br from-zinc-900/90 to-zinc-800/90 backdrop-blur-sm border border-purple-500/30 p-8 hover:border-pink-500/50 hover:shadow-2xl hover:shadow-pink-500/20 transition-all duration-500 hover:-translate-y-2 group cursor-pointer grid grid-rows-[auto_auto_1fr_auto_auto_auto] h-full"
                style={{ animationDelay: `${index * 150}ms` }}
                data-testid={`project-card-${project.id}`}
              >
                <div className="text-6xl font-bold text-purple-500/20 mb-4">0{project.id}</div>
                <h3 className="text-2xl font-bold text-white mb-3 group-hover:text-pink-400 transition-colors">
                  {project.title}
                </h3>
                <p className="text-gray-400 mb-4 leading-relaxed line-clamp-3 self-start">
                  {project.cardTagline || project.tagline}
                </p>

                <div className="mb-4 self-end">
                  {project.highlights.slice(0, 2).map((highlight, i) => (
                    <div key={i} className="flex items-center gap-2 text-sm text-pink-300 mb-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-pink-400 shrink-0" />
                      {highlight}
                    </div>
                  ))}
                </div>

                <div className="flex flex-wrap gap-2 mb-6 self-end content-end">
                  {project.tech.slice(0, 4).map((tech, i) => (
                    <Badge key={i} className="bg-blue-500/10 text-blue-300 border-blue-500/30">
                      {tech}
                    </Badge>
                  ))}
                  {project.tech.length > 4 && (
                    <Badge className="bg-blue-500/10 text-blue-300 border-blue-500/30">
                      +{project.tech.length - 4} more
                    </Badge>
                  )}
                </div>

                <div className="text-purple-400 text-sm font-medium flex items-center gap-2 self-end">
                  View Details <ExternalLink className="w-4 h-4" />
                </div>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Skills Section */}
      <section id="skills" className="relative py-24 px-6" data-testid="skills-section">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16 scroll-reveal">
            <p className="text-purple-400 text-sm uppercase tracking-widest mb-4 font-semibold">What I Do</p>
            <h2 className="text-5xl md:text-6xl font-bold text-white mb-6">Skills & Expertise</h2>
            <p className="text-gray-400 text-lg max-w-2xl mx-auto">
              Specialized in building end-to-end AI solutions from research to production deployment
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {skills.map((skill, index) => (
              <Card
                key={index}
                className="scroll-reveal bg-zinc-900/80 backdrop-blur-sm border border-purple-500/30 p-6 hover:border-pink-500/50 hover:shadow-lg hover:shadow-pink-500/20 transition-all duration-300 hover:-translate-y-1"
                style={{ animationDelay: `${index * 100}ms` }}
                data-testid={`skill-card-${index}`}
              >
                <div className="text-pink-400 mb-4">{skill.icon}</div>
                <h3 className="text-xl font-bold text-white mb-4">{skill.category}</h3>
                <div className="flex flex-wrap gap-2">
                  {skill.items.map((item, i) => (
                    <Badge
                      key={i}
                      variant="secondary"
                      className="bg-purple-500/10 text-purple-300 border-purple-500/30 hover:bg-purple-500/20"
                    >
                      {item}
                    </Badge>
                  ))}
                </div>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Experience Section */}
      <section id="experience" className="relative py-24 px-6" data-testid="experience-section">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16 scroll-reveal">
            <p className="text-purple-400 text-sm uppercase tracking-widest mb-4 font-semibold">My Journey</p>
            <h2 className="text-5xl md:text-6xl font-bold text-white mb-6">Experience & Education</h2>
          </div>

          <div className="relative">
            <div className="absolute left-0 md:left-8 top-0 bottom-0 w-px bg-gradient-to-b from-purple-500 via-pink-500 to-blue-500" />

            {experience.map((exp, index) => (
              <div
                key={index}
                className="scroll-reveal relative pl-8 md:pl-24 pb-12 last:pb-0"
                style={{ animationDelay: `${index * 100}ms` }}
                data-testid={`experience-item-${index}`}
              >
                <div className="absolute left-0 md:left-8 -translate-x-1/2 w-4 h-4 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 shadow-lg shadow-pink-500/50" />

                <Card className="bg-zinc-900/80 backdrop-blur-sm border border-purple-500/30 p-6 hover:border-pink-500/50 transition-all">
                  <div className="flex items-start gap-4 mb-4">
                    <div className="text-purple-400 mt-1">{exp.icon}</div>
                    <div className="flex-1">
                      <p className="text-pink-400 text-sm font-semibold mb-2">{exp.period}</p>
                      <h3 className="text-xl font-bold text-white mb-1">{exp.role}</h3>
                      <p className="text-gray-400 mb-2">{exp.company} • {exp.location}</p>
                    </div>
                  </div>
                  {exp.highlights.length > 0 && (
                    <ul className="space-y-2">
                      {exp.highlights.map((highlight, i) => (
                        <li key={i} className="text-gray-400 text-sm leading-relaxed flex items-start gap-2">
                          <span className="text-purple-400 mt-1">▸</span>
                          <span>{highlight}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </Card>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="relative py-24 px-6" data-testid="contact-section">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16 scroll-reveal">
            <p className="text-pink-400 text-sm uppercase tracking-widest mb-4 font-semibold">Get In Touch</p>
            <h2 className="text-5xl md:text-6xl font-bold text-white mb-6">Let's Work Together</h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="space-y-6 scroll-reveal">
              <Card className="bg-zinc-900/80 backdrop-blur-sm border border-purple-500/30 p-6 hover:border-pink-500/50 transition-all" data-testid="contact-email">
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-lg bg-purple-500/10">
                    <Mail className="w-6 h-6 text-purple-400" />
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm">Email</p>
                    <a
                      href="mailto:aryangupta.7263@gmail.com"
                      rel="me"
                      onClick={() => trackEvent('social_link_clicked', { event_category: 'Contact', event_label: 'Email' })}
                      className="text-white hover:text-purple-400 transition-colors"
                    >
                      aryangupta.7263@gmail.com
                    </a>
                  </div>
                </div>
              </Card>

              <Card className="bg-zinc-900/80 backdrop-blur-sm border border-purple-500/30 p-6 hover:border-pink-500/50 transition-all" data-testid="contact-location">
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-lg bg-blue-500/10">
                    <MapPin className="w-6 h-6 text-blue-400" />
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm">Location</p>
                    <p className="text-white">New Delhi, India</p>
                  </div>
                </div>
              </Card>

              <Card className="bg-zinc-900/80 backdrop-blur-sm border border-purple-500/30 p-6 hover:border-pink-500/50 transition-all" data-testid="contact-linkedin">
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-lg bg-purple-500/10">
                    <Linkedin className="w-6 h-6 text-purple-400" />
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm">LinkedIn</p>
                    <a
                      href="https://www.linkedin.com/in/aryangupta7263"
                      target="_blank"
                      rel="noopener noreferrer me"
                      onClick={() => trackEvent('social_link_clicked', { event_category: 'Contact', event_label: 'LinkedIn' })}
                      className="text-white hover:text-purple-400 transition-colors"
                    >
                      linkedin.com/in/aryangupta7263
                    </a>
                  </div>
                </div>
              </Card>
            </div>

            <Card className="scroll-reveal bg-zinc-900/80 backdrop-blur-sm border border-purple-500/30 p-8" data-testid="contact-form">
              <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                  <input
                    type="text"
                    name="name"
                    placeholder="Your Name"
                    required
                    className="w-full px-4 py-3 bg-black/50 border border-purple-500/30 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/50 transition-all"
                    data-testid="form-name"
                  />
                </div>
                <div>
                  <input
                    type="email"
                    name="email"
                    placeholder="Your Email"
                    required
                    className="w-full px-4 py-3 bg-black/50 border border-purple-500/30 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/50 transition-all"
                    data-testid="form-email"
                  />
                </div>
                <div>
                  <textarea
                    name="message"
                    placeholder="Your Message"
                    rows="5"
                    required
                    className="w-full px-4 py-3 bg-black/50 border border-purple-500/30 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-pink-500/50 focus:ring-1 focus:ring-pink-500/50 transition-all resize-none"
                    data-testid="form-message"
                  />
                </div>
                <Button
                  type="submit"
                  disabled={formStatus === 'sending'}
                  className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white py-3 rounded-lg shadow-lg hover:shadow-purple-500/50 transition-all flex items-center justify-center gap-2 font-bold tracking-wide"
                  data-testid="form-submit"
                >
                  {formStatus === 'sending' ? (
                    'Sending...'
                  ) : formStatus === 'success' ? (
                    'Inquiry Received. Thanks for reaching out.'
                  ) : formStatus === 'error' ? (
                    '❌ Sending Failed'
                  ) : (
                    <>
                      <Send className="w-4 h-4" />
                      Deploy Inquiry
                    </>
                  )}
                </Button>
              </form>
            </Card>
          </div>
        </div>
      </section>

      </main>
      {/* Footer */}
      <footer className="relative py-8 px-6 border-t border-white/5 contentinfo" data-testid="footer">
        <div className="max-w-7xl mx-auto text-center">
          <p className="text-gray-500">
            © 2026 Aryan Gupta. Built with React & Tailwind CSS.
          </p>
        </div>
      </footer>
    </div>
  );
}

function ProjectDetail() {
  const { slug } = useParams();
  const navigate = useNavigate();
  const project = projectsData[slug];
  const [showArchModal, setShowArchModal] = useState(false);

  useEffect(() => {
    window.scrollTo(0, 0);
    if (project) {
      updateSEO({
        title: `${project.title} | Aryan Gupta – AI Engineer`,
        description: `${project.tagline}. Built with ${project.tech.slice(0, 5).join(', ')}. ${project.description.split('\\n\\n')[0].slice(0, 120)}...`,
        canonical: `https://aryangupta.work/project/${project.slug}`,
      });
      
      const script = document.createElement('script');
      script.type = 'application/ld+json';
      script.id = 'project-jsonld';
      script.innerHTML = JSON.stringify({
        "@context": "https://schema.org",
        "@graph": [
          {
            "@type": "SoftwareApplication",
            "name": project.title,
            "url": `https://aryangupta.work/project/${project.slug}`,
            "description": project.tagline,
            "applicationCategory": project.category,
            "operatingSystem": "Web",
            "author": {
              "@type": "Person",
              "name": "Aryan Gupta",
              "url": "https://aryangupta.work/"
            },
            "programmingLanguage": project.tech
          },
          {
            "@type": "BreadcrumbList",
            "itemListElement": [
              { "@type": "ListItem", "position": 1, "name": "Home", "item": "https://aryangupta.work/" },
              { "@type": "ListItem", "position": 2, "name": "Projects", "item": "https://aryangupta.work/#projects" },
              { "@type": "ListItem", "position": 3, "name": project.title, "item": `https://aryangupta.work/project/${project.slug}` }
            ]
          }
        ]
      });
      document.head.appendChild(script);
      
      return () => {
        const existingScript = document.head.querySelector('#project-jsonld');
        if (existingScript) {
          document.head.removeChild(existingScript);
        }
      };
    }
  }, [project]);

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
      <div className="min-h-screen flex items-center justify-center bg-black">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-white mb-4">Project Not Found</h1>
          <Button onClick={() => navigate('/')} className="bg-gradient-to-r from-purple-500 to-pink-500">
            Go Back Home
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="App min-h-screen">
      <main role="main">
      {/* Background */}
      <div className="fixed inset-0 z-0 overflow-hidden">
        <div className="absolute inset-0 bg-black" />
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse-slow" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-pink-500/20 rounded-full blur-3xl animate-pulse-slower" />
        <div className="grid-overlay" />
      </div>

      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-black/80 backdrop-blur-lg border-b border-purple-500/20">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <button
            onClick={() => navigate('/')}
            className="text-2xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent hover:scale-105 transition-transform"
          >
            ARYAN<span className="text-white/50 font-light">.ai</span>
          </button>
          <button
            onClick={handleBackClick}
            className="flex items-center gap-2 text-purple-400 hover:text-pink-400 transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
            Back to Portfolio
          </button>
        </div>
      </nav>

      {/* Project Hero */}
      <section className="relative pt-32 pb-8 px-6">
        <div className="max-w-5xl mx-auto text-center z-10 relative">
          <div className="mb-6 flex items-center justify-center gap-4 flex-wrap">
            <Badge className="bg-purple-500/20 text-purple-300 border-purple-500/50 px-4 py-1">
              {project.category}
            </Badge>
            <Badge className="bg-pink-500/20 text-pink-300 border-pink-500/50 px-4 py-1">
              {project.year}
            </Badge>
          </div>
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
            {project.title}
          </h1>
          <p className="text-xl md:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto">
            {project.tagline}
          </p>

          {/* GitHub, Live Demo and Architecture Buttons */}
          <div className="flex items-center justify-center gap-4 flex-wrap mb-4">
            <a
              href={project.github}
              target="_blank"
              rel="noopener noreferrer"
              onClick={() => trackEvent('project_github_clicked', {
                event_category: 'Projects',
                project_name: project.title,
                project_slug: project.slug,
                event_label: 'View Code',
              })}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white text-base rounded-lg font-semibold transition-all shadow-lg hover:shadow-purple-500/50 hover:scale-105"
              data-testid="project-github-button"
            >
              <Github className="w-5 h-5" />
              View Code
            </a>
            <a
              href={project.demo}
              target="_blank"
              rel="noopener noreferrer"
              onClick={() => trackEvent('project_demo_clicked', {
                event_category: 'Projects',
                project_name: project.title,
                project_slug: project.slug,
                event_label: 'Live Demo',
              })}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white text-base rounded-lg font-semibold transition-all shadow-lg hover:shadow-pink-500/50 hover:scale-105"
              data-testid="project-demo-button"
            >
              <ExternalLink className="w-5 h-5" />
              Live Demo
            </a>
            <button
              onClick={() => {
                trackEvent('project_architecture_clicked', {
                  event_category: 'Projects',
                  project_name: project.title,
                  project_slug: project.slug,
                  event_label: 'Architecture',
                });
                setShowArchModal(true);
              }}
              className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white text-base rounded-lg font-semibold transition-all shadow-lg hover:shadow-purple-500/50 hover:scale-105"
              data-testid="project-architecture-button"
            >
              <Layers className="w-5 h-5" />
              Architecture
            </button>
          </div>
        </div>
      </section>

      {/* Project Content */}
      <section className="relative pt-8 pb-16 px-6">
        <div className="max-w-5xl mx-auto z-10 relative">
          {/* Description */}
          <Card className="bg-zinc-900/80 backdrop-blur-sm border border-purple-500/30 p-8 mb-8">
            <h2 className="text-3xl font-bold text-white mb-6 border-l-4 border-purple-500 pl-4">Overview</h2>
            {project.description.split('\n\n').map((para, i) => (
              <p key={i} className="text-gray-300 text-lg leading-relaxed mb-4 last:mb-0">
                {para}
              </p>
            ))}
          </Card>

          {/* Key Features */}
          <Card className="bg-zinc-900/80 backdrop-blur-sm border border-purple-500/30 p-8 mb-8">
            <h2 className="text-3xl font-bold text-white mb-6 border-l-4 border-pink-500 pl-4">Key Features</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {project.features.map((feature, i) => (
                <Card key={i} className="bg-black/50 border border-purple-500/20 p-6 hover:border-pink-500/50 transition-all">
                  <h3 className="text-xl font-bold text-pink-400 mb-3">{feature.title}</h3>
                  <p className="text-gray-400 leading-relaxed">{feature.desc}</p>
                </Card>
              ))}
            </div>
          </Card>

          {/* Tech Stack */}
          <Card className="bg-zinc-900/80 backdrop-blur-sm border border-purple-500/30 p-8">
            <h2 className="text-3xl font-bold text-white mb-6 border-l-4 border-blue-500 pl-4">Tech Stack</h2>
            <div className="flex flex-wrap gap-3">
              {project.tech.map((tech, i) => (
                <Badge key={i} className="bg-blue-500/10 text-blue-300 border-blue-500/30 px-4 py-2 text-base">
                  {tech}
                </Badge>
              ))}
            </div>
          </Card>

          {/* Back to Portfolio Button */}
          <div className="mt-12 flex justify-center">
            <button
              onClick={handleBackClick}
              className="flex items-center gap-2 px-6 py-3 bg-zinc-800 hover:bg-zinc-700 text-white border border-purple-500/30 rounded-lg transition-all shadow-lg hover:shadow-purple-500/50"
            >
              <ArrowLeft className="w-5 h-5" />
              Back to Portfolio
            </button>
          </div>
        </div>
      </section>

      </main>
      {/* Footer */}
      <footer className="relative py-8 px-6 border-t border-white/5 mt-16 contentinfo">
        <div className="max-w-7xl mx-auto text-center">
          <p className="text-gray-500">© 2026 Aryan Gupta. Built with React & Tailwind CSS.</p>
        </div>
      </footer>

      {/* Architecture Modal */}
      {showArchModal && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 animate-in fade-in duration-200"
          onClick={() => setShowArchModal(false)}
        >
          <div
            className="relative max-w-5xl w-full max-h-[90vh] flex flex-col items-center animate-in zoom-in-95 duration-200"
            onClick={e => e.stopPropagation()}
          >
            <button
              className="absolute -top-12 right-0 text-white/70 hover:text-white transition-colors"
              onClick={() => setShowArchModal(false)}
            >
              <X className="w-8 h-8" />
            </button>
            <div className="bg-zinc-900 border border-purple-500/30 rounded-xl p-2 w-full h-full overflow-hidden shadow-2xl shadow-purple-500/20">
              {project.architecture ? (
                <img
                  src={project.architecture}
                  alt={`${project.title} Architecture`}
                  className="w-full h-full object-contain max-h-[85vh] rounded-lg"
                />
              ) : (
                <div className="w-full h-64 md:h-96 flex flex-col items-center justify-center text-gray-400">
                  <Layers className="w-16 h-16 mb-4 text-purple-500/50" />
                  <p className="text-xl font-semibold text-white">Architecture Diagram</p>
                  <p className="text-sm mt-2 text-center max-w-sm">Detailed architecture layout for this project will be added soon.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

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
