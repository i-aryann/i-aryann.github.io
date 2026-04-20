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
  Layers,
  Network,
  Workflow,
  Server
} from 'lucide-react';

// ─── Project Data ────────────────────────────────────────────────────────────
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
    architecture: '/regulatory_rag.png',
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
    architecture: '',
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
    architecture: '',
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

// ─── Loading component ───────────────────────────────────────────────────────
function PageLoader() {
  return (
    <div className="fixed inset-0 bg-[#F5F5F0] flex items-center justify-center z-50">
      <div className="text-center">
        <div className="w-12 h-12 border-3 border-gray-800 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
        <p className="text-gray-500 text-sm">Loading...</p>
      </div>
    </div>
  );
}

// ─── Page transition wrapper ─────────────────────────────────────────────────
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

// ─── SEO helper ──────────────────────────────────────────────────────────────
const updateSEO = ({ title, description, canonical }) => {
  document.title = title;
  const metaDesc = document.querySelector('meta[name="description"]');
  if (metaDesc) metaDesc.setAttribute('content', description);
  const url = canonical || 'https://aryangupta.work/';
  let canonicalEl = document.querySelector('link[rel="canonical"]');
  if (canonicalEl) canonicalEl.setAttribute('href', url);
  const ogUrl = document.querySelector('meta[property="og:url"]');
  const ogTitle = document.querySelector('meta[property="og:title"]');
  const ogDesc = document.querySelector('meta[property="og:description"]');
  if (ogUrl) ogUrl.setAttribute('content', url);
  if (ogTitle) ogTitle.setAttribute('content', title);
  if (ogDesc) ogDesc.setAttribute('content', description);
  const twTitle = document.querySelector('meta[name="twitter:title"]');
  const twDesc = document.querySelector('meta[name="twitter:description"]');
  if (twTitle) twTitle.setAttribute('content', title);
  if (twDesc) twDesc.setAttribute('content', description);
};

// ─── GA4 tracking helper ─────────────────────────────────────────────────────
const trackEvent = (eventName, params = {}) => {
  if (window.gtag) {
    window.gtag('event', eventName, params);
  }
};

// ─── Portfolio (main page) ───────────────────────────────────────────────────
function Portfolio() {
  const [activeSection, setActiveSection] = useState('home');
  const [isScrolled, setIsScrolled] = useState(false);
  const [formStatus, setFormStatus] = useState('');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const navigate = useNavigate();
  const sectionViewedRef = useRef({});

  // Scroll listener — active section + sticky nav
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

  // Set page SEO on mount
  useEffect(() => {
    updateSEO({
      title: 'Aryan Gupta | AI Engineer · LLM · RAG · MLOps Portfolio',
      description:
        'Aryan Gupta — AI Engineer specialising in LLMs, Retrieval-Augmented Generation (RAG), MLOps, and end-to-end machine-learning systems. Explore projects in NLP, Deep Learning, AWS SageMaker, and production ML pipelines.',
      canonical: 'https://aryangupta.work/',
    });
  }, []);

  // Scroll reveal observer
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

  // Track section views (scroll depth)
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

  // Time-on-page engagement milestones
  useEffect(() => {
    const t30 = setTimeout(() => trackEvent('engaged_30s', { event_category: 'Engagement' }), 30000);
    const t60 = setTimeout(() => trackEvent('engaged_60s', { event_category: 'Engagement' }), 60000);
    const t3m = setTimeout(() => trackEvent('engaged_3min', { event_category: 'Engagement' }), 180000);
    return () => { clearTimeout(t30); clearTimeout(t60); clearTimeout(t3m); };
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
      window.scrollTo({ top: offsetPosition, behavior: 'smooth' });
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
      const response = await fetch('https://formspree.io/f/xqeynyyb', {
        method: 'POST',
        body: data,
        headers: { 'Accept': 'application/json' }
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

  // ─── Capability strip data ─────────────────────────────────────────────────
  const capabilities = [
    { name: 'Multi Agent Agentic Systems', icon: <Network className="w-7 h-7" /> },
    { name: 'Generative AI', icon: <Sparkles className="w-7 h-7" /> },
    { name: 'Advanced RAG', icon: <Database className="w-7 h-7" /> },
    { name: 'LLM Fine-Tuning', icon: <Brain className="w-7 h-7" /> },
    { name: 'API Design', icon: <Workflow className="w-7 h-7" /> },
    { name: 'Cloud AWS', icon: <Cloud className="w-7 h-7" /> },
    { name: 'MLOps', icon: <Server className="w-7 h-7" /> },
  ];

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
        {/* Grid overlay background */}
        <div className="grid-overlay" />

        {/* ════════════════ Navigation ════════════════ */}
        <nav
          className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${isScrolled
            ? 'bg-white/90 backdrop-blur-md shadow-sm border-b border-gray-200'
            : 'bg-white/70 backdrop-blur-sm'
            }`}
          data-testid="main-navigation"
        >
          <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between relative">
            {/* Logo */}
            <button
              onClick={() => navigate('/')}
              className="text-2xl font-bold text-gray-900 hover:opacity-80 transition-opacity"
              data-testid="logo-button"
            >
              ARYAN<span className="text-gray-400 font-light">.ai</span>
            </button>

            {/* Desktop center links */}
            <div className="hidden lg:flex absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 items-center gap-8">
              {['home', 'projects', 'skills', 'experience'].map((section) => (
                <button
                  key={section}
                  onClick={() => {
                    trackEvent('nav_click', { event_category: 'Navigation', section_name: section });
                    navigate('/');
                    setTimeout(() => scrollToSection(section), 100);
                  }}
                  className={`capitalize text-sm font-medium transition-colors ${activeSection === section
                    ? 'text-gray-900'
                    : 'text-gray-500 hover:text-gray-800'
                    }`}
                  data-testid={`nav-${section}`}
                >
                  {section}
                </button>
              ))}
            </div>

            {/* Desktop right buttons */}
            <div className="hidden md:flex items-center gap-3 ml-auto">
              <button
                onClick={() => {
                  trackEvent('nav_click', { event_category: 'Navigation', section_name: 'contact' });
                  navigate('/');
                  setTimeout(() => scrollToSection('contact'), 100);
                }}
                className="px-5 py-2 rounded-full border border-gray-300 bg-white text-gray-800 hover:bg-gray-50 text-sm font-medium transition-all"
                data-testid="nav-contact"
              >
                Start a Conversation
              </button>
              <a
                href={`/Resume.pdf?t=${Date.now()}`}
                target="_blank"
                rel="noopener noreferrer"
                onClick={() => trackEvent('resume_viewed', { event_category: 'Resume', event_label: 'Desktop Nav' })}
                className="px-5 py-2 rounded-full bg-gray-800 text-white hover:bg-gray-900 text-sm font-medium transition-all"
                data-testid="resume-button"
              >
                Get My Resume
              </a>
            </div>

            {/* Mobile menu button */}
            <button
              className="md:hidden text-gray-700 z-50"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              data-testid="mobile-menu-button"
              aria-label="Toggle menu"
            >
              {mobileMenuOpen ? (
                <X className="w-6 h-6" />
              ) : (
                <div className="space-y-1.5">
                  <div className="w-6 h-0.5 bg-gray-700" />
                  <div className="w-6 h-0.5 bg-gray-700" />
                  <div className="w-6 h-0.5 bg-gray-700" />
                </div>
              )}
            </button>
          </div>

          {/* Mobile menu overlay */}
          {mobileMenuOpen && (
            <div className="md:hidden fixed inset-0 top-16 bg-white/95 backdrop-blur-lg z-40 mobile-menu-slide">
              <div className="flex flex-col items-center justify-center h-full gap-8 px-6">
                {['home', 'projects', 'skills', 'experience'].map((section) => (
                  <button
                    key={section}
                    onClick={() => handleMobileNavClick(section)}
                    className={`capitalize text-2xl font-medium transition-colors ${activeSection === section ? 'text-gray-900' : 'text-gray-400 hover:text-gray-700'
                      }`}
                  >
                    {section}
                  </button>
                ))}
                <button
                  onClick={() => handleMobileNavClick('contact')}
                  className="px-8 py-3 rounded-full border border-gray-300 bg-white text-gray-800 text-lg font-medium w-full max-w-xs"
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
                  className="px-8 py-3 rounded-full bg-gray-800 text-white text-lg font-medium text-center w-full max-w-xs"
                >
                  Get My Resume
                </a>
              </div>
            </div>
          )}
        </nav>

        {/* ════════════════ Hero Section (split layout) ════════════════ */}
        <section id="home" className="relative min-h-screen flex items-center px-6 pt-24 pb-12" data-testid="hero-section">
          <div className="max-w-7xl mx-auto w-full z-10">
            <div className="flex flex-col lg:flex-row items-center lg:items-start gap-8 lg:gap-16">
              {/* Left — Avatar */}
              <div className="flex-shrink-0 scroll-reveal">
                <div className="w-48 md:w-64 lg:w-72 relative drop-shadow-[0_15px_30px_rgba(0,0,0,0.15)] transition-transform hover:scale-105 duration-500 ease-out">
                  <img src="/profile.png" alt="Aryan" className="w-full h-auto object-contain rounded-full border-[5px] border-gray-400" />
                </div>
              </div>

              {/* Right — Intro text */}
              <div className="flex-1 scroll-reveal text-center lg:text-left mt-4 lg:mt-6">
                <h1 className="text-5xl md:text-7xl font-black text-gray-900 mb-6 leading-tight tracking-tight">
                  Hi, I'm Aryan
                </h1>
                <p className="text-xl md:text-2xl text-gray-600 mb-2 leading-relaxed max-w-xl">
                  I architect Generative and Agentic AI systems
                  <br />
                  <span className="text-gray-500">for scalable enterprise solutions.</span>
                </p>
                <p className="text-base md:text-lg text-gray-400 mb-10">
                  • LLM Fine-Tuning • Advanced RAG • Scalable Cloud MLOps
                </p>
                <div className="flex flex-wrap justify-center lg:justify-start gap-4 mt-8">
                  <button
                    onClick={() => {
                      trackEvent('cta_click', { event_category: 'Hero', event_label: 'View Projects' });
                      scrollToSection('projects');
                    }}
                    className="bg-gray-800 hover:bg-gray-900 text-white px-8 py-3.5 rounded-full text-sm font-semibold transition-all hover:shadow-lg ring-1 ring-gray-900/10 ring-offset-4 ring-offset-[#F4EFE6]"
                    data-testid="view-work-button"
                  >
                    View Projects
                  </button>
                  <button
                    onClick={() => {
                      trackEvent('cta_click', { event_category: 'Hero', event_label: 'Get in touch' });
                      scrollToSection('contact');
                    }}
                    className="border border-gray-300 bg-white hover:bg-gray-50 text-gray-800 px-8 py-3.5 rounded-full text-sm font-semibold transition-all"
                    data-testid="contact-button"
                  >
                    Get in touch
                  </button>
                </div>
              </div>

              {/* Right — Featured Projects preview */}
            </div>

            {/* Capability strip */}
            <div className="mt-16 flex flex-wrap justify-center gap-3 md:gap-4 scroll-reveal">
              {capabilities.map((cap, i) => (
                <div
                  key={i}
                  className="flex-1 min-w-[calc(50%-0.375rem)] md:min-w-[160px] max-w-[200px] bg-white rounded-2xl p-5 border border-gray-200 shadow-sm text-center hover:shadow-md transition-all flex flex-col items-center justify-center"
                >
                  <div className="text-gray-500 mb-3 flex justify-center">{cap.icon}</div>
                  <p className="text-sm font-medium text-gray-700 leading-snug">{cap.name}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ════════════════ Projects Section ════════════════ */}
        <section id="projects" className="relative py-24 px-6" data-testid="projects-section">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-16 scroll-reveal">
              <p className="text-gray-400 text-sm uppercase tracking-widest mb-3 font-medium">MY WORK</p>
              <h2 className="text-4xl md:text-5xl font-bold text-gray-900">Featured Projects</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {projects.map((project, index) => (
                <div
                  key={project.id}
                  onClick={() => {
                    trackEvent('project_card_clicked', {
                      event_category: 'Projects',
                      project_name: project.title,
                      project_slug: project.slug,
                    });
                    navigate(`/project/${project.slug}`);
                  }}
                  className="scroll-reveal card-hover bg-white border border-gray-200 rounded-2xl p-8 cursor-pointer flex flex-col shadow-sm"
                  data-testid={`project-card-${project.id}`}
                >
                  <div className="text-7xl font-bold text-gray-200 mb-4 leading-none select-none">
                    0{project.id}
                  </div>
                  <h3 className="text-2xl font-bold text-gray-900 mb-3">{project.title}</h3>
                  <p className="text-gray-500 mb-4 leading-relaxed text-sm flex-grow-0">
                    {project.cardTagline || project.tagline}
                  </p>

                  <div className="mb-4">
                    {project.highlights.slice(0, 2).map((h, i) => (
                      <div key={i} className="text-gray-600 text-sm mb-1.5 flex items-start gap-2">
                        <span className="mt-1 text-gray-400">•</span>
                        <span>{h}</span>
                      </div>
                    ))}
                  </div>

                  <div className="flex flex-wrap gap-2 mb-6">
                    {project.tech.slice(0, 4).map((tech, i) => (
                      <span
                        key={i}
                        className="bg-gray-700 text-white text-xs px-3 py-1 rounded-full font-medium"
                      >
                        {tech}
                      </span>
                    ))}
                    {project.tech.length > 4 && (
                      <span className="bg-gray-200 text-gray-600 text-xs px-3 py-1 rounded-full font-medium">
                        +{project.tech.length - 4} more
                      </span>
                    )}
                  </div>

                  <button className="inline-flex items-center gap-2 bg-gray-800 text-white px-5 py-2 rounded-full text-sm font-medium self-start hover:bg-gray-900 transition-colors mt-auto ring-1 ring-gray-900/10 ring-offset-2 ring-offset-white">
                    Live Demo <span>→</span>
                  </button>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ════════════════ Skills Section ════════════════ */}
        <section id="skills" className="relative py-24 px-6" data-testid="skills-section">
          <div className="max-w-7xl mx-auto">
            <div className="text-center mb-16 scroll-reveal">
              <h2 className="text-4xl md:text-5xl font-bold text-gray-900">Skills & Expertise</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {skills.map((skill, index) => (
                <div
                  key={index}
                  className="scroll-reveal bg-white border border-gray-200 rounded-2xl p-8 text-left shadow-sm hover:shadow-md transition-all flex flex-col items-start"
                  data-testid={`skill-card-${index}`}
                >
                  <div className="text-gray-800 mb-5">{skill.icon}</div>
                  <h3 className="text-xl font-bold text-gray-900 mb-4">{skill.category}</h3>
                  <div className="flex flex-wrap gap-2.5">
                    {skill.items.map((item, i) => (
                      <span
                        key={i}
                        className="bg-gray-800 text-white text-xs px-3.5 py-1.5 rounded-lg font-medium tracking-wide"
                      >
                        {item}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ════════════════ Experience Section ════════════════ */}
        <section id="experience" className="relative py-24 px-6" data-testid="experience-section">
          <div className="max-w-4xl mx-auto">
            <div className="text-center mb-16 scroll-reveal">
              <h2 className="text-4xl md:text-5xl font-bold text-gray-900">My Journey</h2>
            </div>

            <div className="relative">
              {/* Timeline line */}
              <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-gray-300" />

              {experience.map((exp, index) => (
                <div
                  key={index}
                  className="scroll-reveal relative pl-16 pb-12 last:pb-0"
                  data-testid={`experience-item-${index}`}
                >
                  {/* Timeline dot */}
                  <div className="absolute left-6 -translate-x-1/2 w-3.5 h-3.5 rounded-full bg-gray-700 border-2 border-[#F5F5F0] z-10" />

                  <div>
                    <p className="text-gray-400 text-sm mb-1">{exp.period}</p>
                    <h3 className="text-lg font-bold text-gray-900 mb-0.5">{exp.role}</h3>
                    <p className="text-gray-500 mb-3">{exp.company} • {exp.location}</p>
                    {exp.highlights.length > 0 && (
                      <ul className="space-y-1.5">
                        {exp.highlights.map((highlight, i) => (
                          <li key={i} className="text-gray-500 text-sm leading-relaxed flex items-start gap-2">
                            <span className="text-gray-400 mt-0.5">•</span>
                            <span>{highlight}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ════════════════ Contact Section ════════════════ */}
        <section id="contact" className="relative py-24 px-6" data-testid="contact-section">
          <div className="max-w-5xl mx-auto">
            <div className="text-center mb-16 scroll-reveal">
              <h2 className="text-4xl md:text-5xl font-bold text-gray-900">Let's Work Together</h2>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Contact info */}
              <div className="space-y-4 scroll-reveal">
                <div className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm hover:shadow-md transition-all" data-testid="contact-email">
                  <div className="flex items-center gap-4">
                    <div className="p-3 rounded-xl bg-gray-100">
                      <Mail className="w-5 h-5 text-gray-600" />
                    </div>
                    <div>
                      <p className="text-gray-400 text-xs uppercase tracking-wide mb-0.5">Email</p>
                      <a
                        href="mailto:aryangupta.7263@gmail.com"
                        rel="me"
                        onClick={() => trackEvent('social_link_clicked', { event_category: 'Contact', event_label: 'Email' })}
                        className="text-gray-800 hover:text-gray-600 transition-colors text-sm font-medium"
                      >
                        aryangupta.7263@gmail.com
                      </a>
                    </div>
                  </div>
                </div>

                <div className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm hover:shadow-md transition-all" data-testid="contact-location">
                  <div className="flex items-center gap-4">
                    <div className="p-3 rounded-xl bg-gray-100">
                      <MapPin className="w-5 h-5 text-gray-600" />
                    </div>
                    <div>
                      <p className="text-gray-400 text-xs uppercase tracking-wide mb-0.5">Location</p>
                      <p className="text-gray-800 text-sm font-medium">New Delhi, India</p>
                    </div>
                  </div>
                </div>

                <div className="bg-white border border-gray-200 rounded-2xl p-6 shadow-sm hover:shadow-md transition-all" data-testid="contact-linkedin">
                  <div className="flex items-center gap-4">
                    <div className="p-3 rounded-xl bg-gray-100">
                      <Linkedin className="w-5 h-5 text-gray-600" />
                    </div>
                    <div>
                      <p className="text-gray-400 text-xs uppercase tracking-wide mb-0.5">LinkedIn</p>
                      <a
                        href="https://www.linkedin.com/in/aryangupta7263"
                        target="_blank"
                        rel="noopener noreferrer me"
                        onClick={() => trackEvent('social_link_clicked', { event_category: 'Contact', event_label: 'LinkedIn' })}
                        className="text-gray-800 hover:text-gray-600 transition-colors text-sm font-medium"
                      >
                        linkedin.com/in/aryangupta7263
                      </a>
                    </div>
                  </div>
                </div>
              </div>

              {/* Contact form */}
              <div className="scroll-reveal bg-white border border-gray-200 rounded-2xl p-8 shadow-sm" data-testid="contact-form">
                <form onSubmit={handleSubmit} className="space-y-5">
                  <div>
                    <input
                      type="text"
                      name="name"
                      placeholder="Your Name"
                      required
                      className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl text-gray-900 placeholder-gray-400 focus:outline-none focus:border-gray-400 focus:ring-1 focus:ring-gray-200 transition-all text-sm"
                      data-testid="form-name"
                    />
                  </div>
                  <div>
                    <input
                      type="email"
                      name="email"
                      placeholder="Your Email"
                      required
                      className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl text-gray-900 placeholder-gray-400 focus:outline-none focus:border-gray-400 focus:ring-1 focus:ring-gray-200 transition-all text-sm"
                      data-testid="form-email"
                    />
                  </div>
                  <div>
                    <textarea
                      name="message"
                      placeholder="Your Message"
                      rows="5"
                      required
                      className="w-full px-4 py-3 bg-gray-50 border border-gray-200 rounded-xl text-gray-900 placeholder-gray-400 focus:outline-none focus:border-gray-400 focus:ring-1 focus:ring-gray-200 transition-all resize-none text-sm"
                      data-testid="form-message"
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={formStatus === 'sending'}
                    className="w-full bg-gray-800 hover:bg-gray-900 text-white py-3 rounded-full text-sm font-semibold transition-all flex items-center justify-center gap-2"
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
                        Send Message
                      </>
                    )}
                  </button>
                </form>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="relative py-8 px-6 border-t border-gray-200 contentinfo" data-testid="footer">
        <div className="max-w-7xl mx-auto text-center">
          <p className="text-gray-400 text-sm">
            © 2026 Aryan Gupta. Built with React & Tailwind CSS.
          </p>
        </div>
      </footer>
    </div>
  );
}

// ─── Project Detail Page ─────────────────────────────────────────────────────
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
        window.scrollTo({ top: offsetPosition, behavior: 'smooth' });
      }
    }, 100);
  };

  if (!project) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#F5F5F0]">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">Project Not Found</h1>
          <button
            onClick={() => navigate('/')}
            className="bg-gray-800 text-white px-6 py-2.5 rounded-full text-sm font-medium hover:bg-gray-900 transition-colors"
          >
            Go Back Home
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="App min-h-screen">
      <main role="main">
        <div className="grid-overlay" />

        {/* Navigation */}
        <nav className="fixed top-0 left-0 right-0 z-50 bg-white/90 backdrop-blur-md border-b border-gray-200 shadow-sm">
          <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <button
              onClick={() => navigate('/')}
              className="text-2xl font-bold text-gray-900 hover:opacity-80 transition-opacity"
            >
              ARYAN<span className="text-gray-400 font-light">.ai</span>
            </button>
            <button
              onClick={handleBackClick}
              className="flex items-center gap-2 text-gray-600 hover:text-gray-900 transition-colors text-sm font-medium"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Portfolio
            </button>
          </div>
        </nav>

        {/* Project Hero */}
        <section className="relative pt-32 pb-8 px-6">
          <div className="max-w-5xl mx-auto text-center z-10 relative">
            <div className="mb-6 flex items-center justify-center gap-3 flex-wrap">
              <span className="bg-gray-700 text-white text-xs px-4 py-1.5 rounded-full font-medium">
                {project.category}
              </span>
              <span className="bg-gray-200 text-gray-700 text-xs px-4 py-1.5 rounded-full font-medium">
                {project.year}
              </span>
            </div>
            <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
              {project.title}
            </h1>
            <p className="text-lg md:text-xl text-gray-500 mb-8 max-w-3xl mx-auto">
              {project.tagline}
            </p>

            {/* Action buttons */}
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
                className="flex items-center gap-2 px-6 py-2.5 bg-gray-800 hover:bg-gray-900 text-white text-sm rounded-full font-medium transition-all"
                data-testid="project-github-button"
              >
                <Github className="w-4 h-4" />
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
                className="flex items-center gap-2 px-6 py-2.5 bg-gray-800 hover:bg-gray-900 text-white text-sm rounded-full font-medium transition-all"
                data-testid="project-demo-button"
              >
                <ExternalLink className="w-4 h-4" />
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
                className="flex items-center gap-2 px-6 py-2.5 border border-gray-300 bg-white hover:bg-gray-50 text-gray-700 text-sm rounded-full font-medium transition-all"
                data-testid="project-architecture-button"
              >
                <Layers className="w-4 h-4" />
                Architecture
              </button>
            </div>
          </div>
        </section>

        {/* Project Content */}
        <section className="relative pt-8 pb-16 px-6">
          <div className="max-w-5xl mx-auto z-10 relative">
            {/* Description */}
            <div className="bg-white border border-gray-200 rounded-2xl p-8 mb-8 shadow-sm">
              <h2 className="text-2xl font-bold text-gray-900 mb-6 border-l-4 border-gray-800 pl-4">Overview</h2>
              {project.description.split('\n\n').map((para, i) => (
                <p key={i} className="text-gray-500 text-base leading-relaxed mb-4 last:mb-0">
                  {para}
                </p>
              ))}
            </div>

            {/* Key Features */}
            <div className="bg-white border border-gray-200 rounded-2xl p-8 mb-8 shadow-sm">
              <h2 className="text-2xl font-bold text-gray-900 mb-6 border-l-4 border-gray-800 pl-4">Key Features</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {project.features.map((feature, i) => (
                  <div key={i} className="bg-gray-50 border border-gray-200 rounded-xl p-6">
                    <h3 className="text-lg font-bold text-gray-900 mb-2">{feature.title}</h3>
                    <p className="text-gray-500 text-sm leading-relaxed">{feature.desc}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Tech Stack */}
            <div className="bg-white border border-gray-200 rounded-2xl p-8 shadow-sm">
              <h2 className="text-2xl font-bold text-gray-900 mb-6 border-l-4 border-gray-800 pl-4">Tech Stack</h2>
              <div className="flex flex-wrap gap-3">
                {project.tech.map((tech, i) => (
                  <span key={i} className="bg-gray-700 text-white text-sm px-4 py-1.5 rounded-full font-medium">
                    {tech}
                  </span>
                ))}
              </div>
            </div>

            {/* Back button */}
            <div className="mt-12 flex justify-center">
              <button
                onClick={handleBackClick}
                className="flex items-center gap-2 px-6 py-2.5 bg-white hover:bg-gray-50 text-gray-700 border border-gray-300 rounded-full text-sm font-medium transition-all"
              >
                <ArrowLeft className="w-4 h-4" />
                Back to Portfolio
              </button>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="relative py-8 px-6 border-t border-gray-200 mt-16 contentinfo">
        <div className="max-w-7xl mx-auto text-center">
          <p className="text-gray-400 text-sm">© 2026 Aryan Gupta. Built with React & Tailwind CSS.</p>
        </div>
      </footer>

      {/* Architecture Modal */}
      {showArchModal && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40 backdrop-blur-sm p-4"
          onClick={() => setShowArchModal(false)}
        >
          <div
            className="relative max-w-5xl w-full max-h-[90vh] flex flex-col items-center"
            onClick={e => e.stopPropagation()}
          >
            <button
              className="absolute -top-12 right-0 text-white/80 hover:text-white transition-colors"
              onClick={() => setShowArchModal(false)}
            >
              <X className="w-8 h-8" />
            </button>
            <div className="bg-white border border-gray-200 rounded-2xl p-2 w-full h-full overflow-hidden shadow-2xl">
              {project.architecture ? (
                <img
                  src={project.architecture}
                  alt={`${project.title} Architecture`}
                  className="w-full h-full object-contain max-h-[85vh] rounded-xl"
                />
              ) : (
                <div className="w-full h-64 md:h-96 flex flex-col items-center justify-center text-gray-400">
                  <Layers className="w-12 h-12 mb-4 text-gray-300" />
                  <p className="text-lg font-semibold text-gray-600">Architecture Diagram</p>
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

// ─── App Router ──────────────────────────────────────────────────────────────
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
