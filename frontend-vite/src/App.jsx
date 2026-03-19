import { useEffect, useState, lazy, Suspense } from 'react';
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
  X
} from 'lucide-react';

// Project Data with full details
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

function Portfolio() {
  const [activeSection, setActiveSection] = useState('home');
  const [isScrolled, setIsScrolled] = useState(false);
  const [formStatus, setFormStatus] = useState('');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const navigate = useNavigate();

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

    window.addEventListener('scroll', handleScroll);
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
    
    setTimeout(() => {
      setFormStatus('success');
      e.target.reset();
      setTimeout(() => setFormStatus(''), 3000);
    }, 1000);
  };

  const skills = [
    {
      category: 'Languages',
      icon: <Code2 className="w-8 h-8" />,
      items: ['Python', 'SQL', 'JavaScript']
    },
    {
      category: 'AI & Machine Learning',
      icon: <Brain className="w-8 h-8" />,
      items: ['TensorFlow', 'PyTorch', 'Scikit-Learn', 'XGBoost', 'Random Forest', 'K-Means']
    },
    {
      category: 'Deep Learning & NLP',
      icon: <Sparkles className="w-8 h-8" />,
      items: ['Transformers', 'BERT', 'LSTMs', 'Hugging Face', 'Keras']
    },
    {
      category: 'MLOps & Cloud',
      icon: <Cloud className="w-8 h-8" />,
      items: ['AWS SageMaker', 'AWS EC2', 'AWS S3', 'AWS Lambda', 'Docker', 'CI/CD Pipelines']
    },
    {
      category: 'Data & Analytics',
      icon: <Database className="w-8 h-8" />,
      items: ['Pandas', 'NumPy', 'Power BI', 'Matplotlib', 'Seaborn', 'Plotly']
    },
    {
      category: 'Databases',
      icon: <Database className="w-8 h-8" />,
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

  return (
    <div className="App">
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
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
          isScrolled ? 'bg-black/80 backdrop-blur-lg border-b border-purple-500/20 shadow-lg shadow-purple-500/5' : 'bg-transparent'
        }`}
        data-testid="main-navigation"
      >
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <button 
            onClick={() => navigate('/')}
            className="text-2xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent hover:scale-105 transition-transform"
            data-testid="logo-button"
          >
            ARYAN<span className="text-white/50 font-light">.ai</span>
          </button>

          {/* Desktop Menu */}
          <div className="hidden md:flex items-center gap-8">
            {['home', 'skills', 'projects', 'experience', 'contact'].map((section) => (
              <button
                key={section}
                onClick={() => { navigate('/'); setTimeout(() => scrollToSection(section), 100); }}
                className={`capitalize text-sm font-medium transition-all ${
                  activeSection === section
                    ? 'text-pink-400'
                    : 'text-gray-400 hover:text-white'
                }`}
                data-testid={`nav-${section}`}
              >
                {section}
              </button>
            ))}
            <a
              href="#resume"
              className="px-4 py-2 rounded-lg border border-purple-500/50 bg-purple-500/10 text-purple-400 hover:bg-purple-500/20 transition-all text-sm font-medium flex items-center gap-2"
              data-testid="resume-button"
            >
              <FileText className="w-4 h-4" />
              Resume
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
              {['home', 'skills', 'projects', 'experience', 'contact'].map((section) => (
                <button
                  key={section}
                  onClick={() => handleMobileNavClick(section)}
                  className={`capitalize text-2xl font-medium transition-all ${
                    activeSection === section
                      ? 'text-pink-400'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  {section}
                </button>
              ))}
              <a
                href="#resume"
                onClick={() => setMobileMenuOpen(false)}
                className="px-6 py-3 rounded-lg border border-purple-500/50 bg-purple-500/10 text-purple-400 text-lg font-medium flex items-center gap-2"
              >
                <FileText className="w-5 h-5" />
                Resume
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
              I build scalable machine learning and artificial intelligence systems with automated MLOps pipelines and cloud-native deployment strategies.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-4">
              <Button
                onClick={() => scrollToSection('projects')}
                className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white px-8 py-6 text-lg rounded-lg shadow-lg hover:shadow-purple-500/50 transition-all"
                data-testid="view-work-button"
              >
                View My Work
              </Button>
              <Button
                onClick={() => scrollToSection('contact')}
                variant="outline"
                className="border-2 border-white/20 bg-white/5 hover:bg-white/10 text-white px-8 py-6 text-lg rounded-lg backdrop-blur-sm"
                data-testid="contact-button"
              >
                Get In Touch
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

      {/* Projects Section */}
      <section id="projects" className="relative py-24 px-6" data-testid="projects-section">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16 scroll-reveal">
            <p className="text-pink-400 text-sm uppercase tracking-widest mb-4 font-semibold">My Work</p>
            <h2 className="text-5xl md:text-6xl font-bold text-white mb-6">Featured Projects</h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-8">
            {projects.map((project, index) => (
              <Card
                key={project.id}
                onClick={() => navigate(`/project/${project.slug}`)}
                className="scroll-reveal bg-gradient-to-br from-zinc-900/90 to-zinc-800/90 backdrop-blur-sm border border-purple-500/30 p-8 hover:border-pink-500/50 hover:shadow-2xl hover:shadow-pink-500/20 transition-all duration-500 hover:-translate-y-2 group cursor-pointer"
                style={{ animationDelay: `${index * 150}ms` }}
                data-testid={`project-card-${project.id}`}
              >
                <div className="text-6xl font-bold text-purple-500/20 mb-4">0{project.id}</div>
                <h3 className="text-2xl font-bold text-white mb-3 group-hover:text-pink-400 transition-colors">
                  {project.title}
                </h3>
                <p className="text-gray-400 mb-4 leading-relaxed line-clamp-3">
                  {project.description.split('\n\n')[0]}
                </p>
                
                <div className="mb-4">
                  {project.highlights.slice(0, 2).map((highlight, i) => (
                    <div key={i} className="flex items-center gap-2 text-sm text-pink-300 mb-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-pink-400" />
                      {highlight}
                    </div>
                  ))}
                </div>

                <div className="flex flex-wrap gap-2 mb-6">
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

                <div className="text-purple-400 text-sm font-medium flex items-center gap-2">
                  View Details <ExternalLink className="w-4 h-4" />
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
                    <a href="mailto:aryangupta.7263@gmail.com" className="text-white hover:text-purple-400 transition-colors">
                      aryangupta.7263@gmail.com
                    </a>
                  </div>
                </div>
              </Card>

              <Card className="bg-zinc-900/80 backdrop-blur-sm border border-purple-500/30 p-6 hover:border-pink-500/50 transition-all" data-testid="contact-phone">
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-lg bg-pink-500/10">
                    <Phone className="w-6 h-6 text-pink-400" />
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm">Phone</p>
                    <a href="tel:+917534090544" className="text-white hover:text-pink-400 transition-colors">
                      +91 7534090544
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
                      rel="noopener noreferrer"
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
                  className="w-full bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white py-3 rounded-lg shadow-lg hover:shadow-purple-500/50 transition-all flex items-center justify-center gap-2"
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
                </Button>
              </form>
            </Card>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative py-8 px-6 border-t border-white/5" data-testid="footer">
        <div className="max-w-7xl mx-auto text-center">
          <p className="text-gray-500">
            © 2025 Aryan Gupta. Built with React & Tailwind CSS.
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
      <section className="relative pt-32 pb-16 px-6">
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
          
          {/* GitHub and Live Demo Buttons */}
          <div className="flex items-center justify-center gap-6 flex-wrap mb-4">
            <a
              href={project.github}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white text-lg rounded-xl font-semibold transition-all shadow-lg hover:shadow-purple-500/50 hover:scale-105"
              data-testid="project-github-button"
            >
              <Github className="w-6 h-6" />
              View Code
            </a>
            <a
              href={project.demo}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-3 px-8 py-4 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white text-lg rounded-xl font-semibold transition-all shadow-lg hover:shadow-pink-500/50 hover:scale-105"
              data-testid="project-demo-button"
            >
              <ExternalLink className="w-6 h-6" />
              Live Demo
            </a>
          </div>
        </div>
      </section>

      {/* Project Content */}
      <section className="relative py-16 px-6">
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
        </div>
      </section>

      {/* Footer */}
      <footer className="relative py-8 px-6 border-t border-white/5 mt-16">
        <div className="max-w-7xl mx-auto text-center">
          <p className="text-gray-500">© 2025 Aryan Gupta. Built with React & Tailwind CSS.</p>
        </div>
      </footer>
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
