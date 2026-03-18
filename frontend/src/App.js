import { useEffect, useState, useRef } from 'react';
import '@/App.css';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
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
  GraduationCap
} from 'lucide-react';

function App() {
  const [activeSection, setActiveSection] = useState('home');
  const [isScrolled, setIsScrolled] = useState(false);
  const [formStatus, setFormStatus] = useState('');

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);

      // Update active section based on scroll position
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

  // Intersection Observer for scroll animations
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
    
    // Simulate form submission (connect to backend later)
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

  const projects = [
    {
      id: 1,
      title: 'Credit Default Risk Analyzer',
      description: 'Built a machine learning model to predict customer credit default with 92% ROC-AUC score using ensemble methods and SHAP for explainability.',
      tech: ['Python', 'XGBoost', 'Scikit-learn', 'Streamlit', 'AWS SageMaker', 'Docker'],
      github: 'https://github.com/i-aryann/Credit-Default-Prediction',
      demo: 'https://credit-default-prediction-aryan.streamlit.app/',
      highlights: ['92% ROC-AUC Score', 'SHAP Integration', 'Automated CI/CD Pipeline']
    },
    {
      id: 2,
      title: 'NLP Sentiment Analyzer',
      description: 'Created a real-time sentiment analysis tool using BERT transformers for social media data with Kafka streaming integration.',
      tech: ['NLP', 'BERT', 'Transformers', 'FastAPI', 'Kafka', 'React'],
      github: '#',
      demo: '#',
      highlights: ['BERT Architecture', 'Real-time Processing', 'Interactive Dashboard']
    },
    {
      id: 3,
      title: 'Sales Forecasting Dashboard',
      description: 'Designed an interactive dashboard with time series forecasting using LSTM networks for revenue prediction and scenario planning.',
      tech: ['Time Series', 'LSTM', 'TensorFlow', 'Streamlit', 'PostgreSQL', 'Plotly'],
      github: '#',
      demo: '#',
      highlights: ['LSTM Networks', 'What-if Scenarios', 'Automated Reporting']
    }
  ];

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
        <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950" />
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl animate-pulse-slow" />
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse-slower" />
        <div className="grid-overlay" />
      </div>

      {/* Sticky Navigation */}
      <nav 
        className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
          isScrolled ? 'bg-slate-950/80 backdrop-blur-lg border-b border-cyan-500/10 shadow-lg' : 'bg-transparent'
        }`}
        data-testid="main-navigation"
      >
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <button 
            onClick={() => scrollToSection('home')} 
            className="text-2xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent hover:scale-105 transition-transform"
            data-testid="logo-button"
          >
            ARYAN<span className="text-white/50 font-light">.ai</span>
          </button>

          <div className="hidden md:flex items-center gap-8">
            {['home', 'skills', 'projects', 'experience', 'contact'].map((section) => (
              <button
                key={section}
                onClick={() => scrollToSection(section)}
                className={`capitalize text-sm font-medium transition-all ${
                  activeSection === section
                    ? 'text-cyan-400'
                    : 'text-gray-400 hover:text-white'
                }`}
                data-testid={`nav-${section}`}
              >
                {section}
              </button>
            ))}
            <a
              href="#resume"
              className="px-4 py-2 rounded-lg border border-emerald-500/50 bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20 transition-all text-sm font-medium flex items-center gap-2"
              data-testid="resume-button"
            >
              <FileText className="w-4 h-4" />
              Resume
            </a>
          </div>

          {/* Mobile Menu Button */}
          <button className="md:hidden text-white" data-testid="mobile-menu-button">
            <div className="w-6 h-0.5 bg-white mb-1" />
            <div className="w-6 h-0.5 bg-white mb-1" />
            <div className="w-6 h-0.5 bg-white" />
          </button>
        </div>
      </nav>

      {/* Hero Section */}
      <section id="home" className="relative min-h-screen flex items-center justify-center px-6" data-testid="hero-section">
        <div className="max-w-5xl mx-auto text-center z-10">
          <div className="scroll-reveal">
            <h1 className="text-6xl md:text-8xl font-bold mb-6 leading-tight">
              Hi, I'm{' '}
              <span className="bg-gradient-to-r from-emerald-400 via-cyan-400 to-blue-500 bg-clip-text text-transparent animate-gradient">
                Aryan
              </span>
            </h1>
            <div className="inline-block mb-8">
              <Badge className="px-4 py-2 text-lg bg-cyan-500/10 border-cyan-500/50 text-cyan-300 hover:bg-cyan-500/20">
                AI Engineer
              </Badge>
            </div>
            <p className="text-xl md:text-2xl text-gray-300 mb-12 max-w-3xl mx-auto leading-relaxed">
              I build scalable machine learning and artificial intelligence systems with automated MLOps pipelines and cloud-native deployment strategies.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-4">
              <Button
                onClick={() => scrollToSection('projects')}
                className="bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-600 hover:to-cyan-600 text-white px-8 py-6 text-lg rounded-lg shadow-lg hover:shadow-emerald-500/50 transition-all"
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

        {/* Scroll Indicator */}
        <button 
          onClick={() => scrollToSection('skills')}
          className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce cursor-pointer z-10"
          data-testid="scroll-indicator"
        >
          <ChevronDown className="w-8 h-8 text-cyan-400" />
        </button>
      </section>

      {/* Skills Section */}
      <section id="skills" className="relative py-24 px-6" data-testid="skills-section">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16 scroll-reveal">
            <p className="text-emerald-400 text-sm uppercase tracking-widest mb-4 font-semibold">What I Do</p>
            <h2 className="text-5xl md:text-6xl font-bold text-white mb-6">Skills & Expertise</h2>
            <p className="text-gray-400 text-lg max-w-2xl mx-auto">
              Specialized in building end-to-end AI solutions from research to production deployment
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {skills.map((skill, index) => (
              <Card
                key={index}
                className="scroll-reveal bg-slate-900/50 backdrop-blur-sm border border-cyan-500/20 p-6 hover:border-cyan-500/50 hover:shadow-lg hover:shadow-cyan-500/10 transition-all duration-300 hover:-translate-y-1"
                style={{ animationDelay: `${index * 100}ms` }}
                data-testid={`skill-card-${index}`}
              >
                <div className="text-cyan-400 mb-4">{skill.icon}</div>
                <h3 className="text-xl font-bold text-white mb-4">{skill.category}</h3>
                <div className="flex flex-wrap gap-2">
                  {skill.items.map((item, i) => (
                    <Badge
                      key={i}
                      variant="secondary"
                      className="bg-emerald-500/10 text-emerald-300 border-emerald-500/20 hover:bg-emerald-500/20"
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
            <p className="text-cyan-400 text-sm uppercase tracking-widest mb-4 font-semibold">My Work</p>
            <h2 className="text-5xl md:text-6xl font-bold text-white mb-6">Featured Projects</h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-8">
            {projects.map((project, index) => (
              <Card
                key={project.id}
                className="scroll-reveal bg-gradient-to-br from-slate-900/90 to-slate-800/90 backdrop-blur-sm border border-emerald-500/20 p-8 hover:border-emerald-500/50 hover:shadow-2xl hover:shadow-emerald-500/20 transition-all duration-500 hover:-translate-y-2 group"
                style={{ animationDelay: `${index * 150}ms` }}
                data-testid={`project-card-${project.id}`}
              >
                <div className="text-6xl font-bold text-emerald-500/20 mb-4">0{project.id}</div>
                <h3 className="text-2xl font-bold text-white mb-3 group-hover:text-emerald-400 transition-colors">
                  {project.title}
                </h3>
                <p className="text-gray-400 mb-4 leading-relaxed">{project.description}</p>
                
                <div className="mb-4">
                  {project.highlights.map((highlight, i) => (
                    <div key={i} className="flex items-center gap-2 text-sm text-cyan-300 mb-2">
                      <div className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
                      {highlight}
                    </div>
                  ))}
                </div>

                <div className="flex flex-wrap gap-2 mb-6">
                  {project.tech.map((tech, i) => (
                    <Badge key={i} className="bg-cyan-500/10 text-cyan-300 border-cyan-500/30">
                      {tech}
                    </Badge>
                  ))}
                </div>

                <div className="flex gap-3">
                  {project.github !== '#' && (
                    <a
                      href={project.github}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors"
                      data-testid={`github-link-${project.id}`}
                    >
                      <Github className="w-4 h-4" />
                      Code
                    </a>
                  )}
                  {project.demo !== '#' && (
                    <a
                      href={project.demo}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-2 text-sm text-emerald-400 hover:text-emerald-300 transition-colors"
                      data-testid={`demo-link-${project.id}`}
                    >
                      <ExternalLink className="w-4 h-4" />
                      Live Demo
                    </a>
                  )}
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
            <p className="text-emerald-400 text-sm uppercase tracking-widest mb-4 font-semibold">My Journey</p>
            <h2 className="text-5xl md:text-6xl font-bold text-white mb-6">Experience & Education</h2>
          </div>

          <div className="relative">
            {/* Timeline Line */}
            <div className="absolute left-0 md:left-8 top-0 bottom-0 w-px bg-gradient-to-b from-emerald-500 via-cyan-500 to-blue-500" />

            {experience.map((exp, index) => (
              <div
                key={index}
                className="scroll-reveal relative pl-8 md:pl-24 pb-12 last:pb-0"
                style={{ animationDelay: `${index * 100}ms` }}
                data-testid={`experience-item-${index}`}
              >
                {/* Timeline Dot */}
                <div className="absolute left-0 md:left-8 -translate-x-1/2 w-4 h-4 rounded-full bg-gradient-to-r from-emerald-500 to-cyan-500 shadow-lg shadow-cyan-500/50" />

                <Card className="bg-slate-900/50 backdrop-blur-sm border border-cyan-500/20 p-6 hover:border-cyan-500/50 transition-all">
                  <div className="flex items-start gap-4 mb-4">
                    <div className="text-emerald-400 mt-1">{exp.icon}</div>
                    <div className="flex-1">
                      <p className="text-cyan-400 text-sm font-semibold mb-2">{exp.period}</p>
                      <h3 className="text-xl font-bold text-white mb-1">{exp.role}</h3>
                      <p className="text-gray-400 mb-2">{exp.company} • {exp.location}</p>
                    </div>
                  </div>
                  {exp.highlights.length > 0 && (
                    <ul className="space-y-2">
                      {exp.highlights.map((highlight, i) => (
                        <li key={i} className="text-gray-400 text-sm leading-relaxed flex items-start gap-2">
                          <span className="text-emerald-400 mt-1">▸</span>
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
            <p className="text-cyan-400 text-sm uppercase tracking-widest mb-4 font-semibold">Get In Touch</p>
            <h2 className="text-5xl md:text-6xl font-bold text-white mb-6">Let's Work Together</h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Contact Info */}
            <div className="space-y-6 scroll-reveal">
              <Card className="bg-slate-900/50 backdrop-blur-sm border border-cyan-500/20 p-6 hover:border-cyan-500/50 transition-all" data-testid="contact-email">
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-lg bg-emerald-500/10">
                    <Mail className="w-6 h-6 text-emerald-400" />
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm">Email</p>
                    <a href="mailto:aryangupta.7263@gmail.com" className="text-white hover:text-emerald-400 transition-colors">
                      aryangupta.7263@gmail.com
                    </a>
                  </div>
                </div>
              </Card>

              <Card className="bg-slate-900/50 backdrop-blur-sm border border-cyan-500/20 p-6 hover:border-cyan-500/50 transition-all" data-testid="contact-phone">
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-lg bg-cyan-500/10">
                    <Phone className="w-6 h-6 text-cyan-400" />
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm">Phone</p>
                    <a href="tel:+917534090544" className="text-white hover:text-cyan-400 transition-colors">
                      +91 7534090544
                    </a>
                  </div>
                </div>
              </Card>

              <Card className="bg-slate-900/50 backdrop-blur-sm border border-cyan-500/20 p-6 hover:border-cyan-500/50 transition-all" data-testid="contact-location">
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

              <Card className="bg-slate-900/50 backdrop-blur-sm border border-cyan-500/20 p-6 hover:border-cyan-500/50 transition-all" data-testid="contact-linkedin">
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-lg bg-emerald-500/10">
                    <Linkedin className="w-6 h-6 text-emerald-400" />
                  </div>
                  <div>
                    <p className="text-gray-400 text-sm">LinkedIn</p>
                    <a 
                      href="https://www.linkedin.com/in/aryangupta7263" 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="text-white hover:text-emerald-400 transition-colors"
                    >
                      linkedin.com/in/aryangupta7263
                    </a>
                  </div>
                </div>
              </Card>
            </div>

            {/* Contact Form */}
            <Card className="scroll-reveal bg-slate-900/50 backdrop-blur-sm border border-emerald-500/20 p-8" data-testid="contact-form">
              <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                  <input
                    type="text"
                    name="name"
                    placeholder="Your Name"
                    required
                    className="w-full px-4 py-3 bg-slate-950/50 border border-cyan-500/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/50 transition-all"
                    data-testid="form-name"
                  />
                </div>
                <div>
                  <input
                    type="email"
                    name="email"
                    placeholder="Your Email"
                    required
                    className="w-full px-4 py-3 bg-slate-950/50 border border-cyan-500/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/50 transition-all"
                    data-testid="form-email"
                  />
                </div>
                <div>
                  <textarea
                    name="message"
                    placeholder="Your Message"
                    rows="5"
                    required
                    className="w-full px-4 py-3 bg-slate-950/50 border border-cyan-500/20 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/50 transition-all resize-none"
                    data-testid="form-message"
                  />
                </div>
                <Button
                  type="submit"
                  disabled={formStatus === 'sending'}
                  className="w-full bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-600 hover:to-cyan-600 text-white py-3 rounded-lg shadow-lg hover:shadow-emerald-500/50 transition-all flex items-center justify-center gap-2"
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

export default App;