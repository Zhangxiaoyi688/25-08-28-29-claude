# Academic Research Portfolio

A professional Jekyll-based website template for academic researchers to showcase their publications, research, and academic profile. This template is designed to be deployed on GitHub Pages with automatic content processing from markdown files.

## ✨ Features

- **📚 Automatic Publication Management**: Add research papers as markdown files in `_research/` folder
- **🎨 Professional Design**: Clean, modern interface with responsive design
- **🔍 Advanced Search & Filtering**: Real-time search and filtering by type, year, and keywords
- **📱 Mobile Responsive**: Optimized for all device sizes
- **🌙 Dark Mode Support**: Toggle between light and dark themes
- **⚡ Fast Performance**: Optimized for speed and SEO
- **🚀 GitHub Pages Ready**: Automatic deployment with GitHub Actions
- **♿ Accessible**: WCAG compliant design

## 🚀 Quick Start

### 1. Use This Template

1. Click "Use this template" on GitHub
2. Create a new repository named `username.github.io` (replace `username` with your GitHub username)
3. Clone your new repository locally

### 2. Customize Your Information

Edit `_config.yml` to update your personal information:

```yaml
# Site settings
title: "Dr. Your Name - Research Portfolio"
description: "Your research description"
url: "https://yourusername.github.io"

# Researcher information
researcher:
  name: "Dr. Your Name"
  title: "Your Title"
  institution: "Your Institution"
  email: "your.email@institution.edu"
  bio: "Your bio here..."
  profile_image: "/assets/images/profile.jpg"
  social_links:
    - name: "Google Scholar"
      url: "https://scholar.google.com/citations?user=your-id"
      icon: "graduation-cap"
    # Add more social links...
```

### 3. Add Your Publications

Create markdown files in the `_research/` folder for each publication:

```markdown
---
title: "Your Paper Title"
authors: ["Author 1", "Author 2"]
date: 2024-01-15
venue: "Conference/Journal Name"
type: "conference" # or "journal", "preprint", "workshop"
tags: ["tag1", "tag2", "tag3"]
abstract: "Your paper abstract..."
pdf_url: "/assets/papers/your-paper.pdf"
external_url: "https://publisher-link.com"
---

# Your paper content in markdown format

## Introduction

Your paper content here...
```

### 4. Update About Page

Edit `pages/about.md` to customize your academic background, experience, and achievements.

### 5. Add Your Profile Image

Add your profile image to `assets/images/profile.jpg` (or update the path in `_config.yml`).

## 📁 Project Structure

```
.
├── _config.yml              # Site configuration
├── _layouts/                 # Page layouts
│   ├── default.html         # Base layout
│   ├── home.html           # Homepage layout
│   ├── page.html           # Static pages layout
│   └── research.html       # Research paper layout
├── _includes/               # Reusable components
│   ├── header.html         # Site header
│   ├── footer.html         # Site footer
│   └── navigation.html     # Navigation components
├── _sass/                   # Stylesheet partials
│   ├── _base.scss          # Base styles and variables
│   ├── _layout.scss        # Layout styles
│   └── _components.scss    # Component styles
├── _research/               # Research papers (markdown files)
│   ├── 2024-01-15-paper1.md
│   └── 2023-12-10-paper2.md
├── assets/                  # Static assets
│   ├── css/
│   ├── js/
│   ├── images/
│   └── papers/             # PDF files
├── pages/                   # Static pages
│   ├── publications.md     # Publications listing
│   └── about.md           # About page
├── .github/workflows/       # GitHub Actions
│   └── deploy.yml          # Deployment workflow
├── index.md                # Homepage
├── Gemfile                 # Ruby dependencies
└── README.md              # This file
```

## 🎨 Customization

### Colors and Styling

Customize the color scheme by editing CSS variables in `_sass/_base.scss`:

```scss
:root {
  --color-primary: #1e3a8a;        # Primary color
  --color-primary-light: #3b82f6;  # Light variant
  --color-accent: #f59e0b;          # Accent color
  // ... more variables
}
```

### Navigation Menu

Update the navigation menu in `_config.yml`:

```yaml
navigation:
  - name: "Home"
    url: "/"
  - name: "Publications"
    url: "/publications/"
  - name: "About"
    url: "/about/"
  - name: "CV"
    url: "/assets/cv.pdf"
```

### Research Categories

Customize publication types and their colors:

```yaml
research_types:
  - name: "Conference Papers"
    type: "conference"
    color: "#3b82f6"
  - name: "Journal Articles"
    type: "journal"
    color: "#10b981"
  # Add more types...
```

## 🚀 Deployment

### GitHub Pages (Recommended)

1. Push your changes to the `main` branch
2. Go to your repository settings
3. Navigate to "Pages" section
4. Select "GitHub Actions" as the source
5. The site will automatically build and deploy

### Local Development

1. Install Ruby and Bundler
2. Run the following commands:

```bash
# Install dependencies
bundle install

# Start local server
bundle exec jekyll serve

# Open http://localhost:4000 in your browser
```

### Custom Domain

To use a custom domain:

1. Add a `CNAME` file to the root directory with your domain name
2. Update the `url` in `_config.yml`
3. Configure DNS settings with your domain provider

## 📝 Content Management

### Adding Publications

1. Create a new markdown file in `_research/`
2. Use the naming convention: `YYYY-MM-DD-title-slug.md`
3. Include proper frontmatter with metadata
4. Write the paper content in markdown
5. Add PDF files to `assets/papers/`

### Publication Frontmatter

```yaml
---
title: "Paper Title"                    # Required
authors: ["Author 1", "Author 2"]      # Required
date: 2024-01-15                        # Required (YYYY-MM-DD)
venue: "Conference/Journal Name"         # Optional
type: "conference"                      # Required (conference, journal, preprint, workshop)
tags: ["tag1", "tag2"]                 # Optional
abstract: "Paper abstract..."           # Optional but recommended
pdf_url: "/assets/papers/paper.pdf"    # Optional
external_url: "https://publisher.com"   # Optional
doi: "10.1000/xyz123"                  # Optional
---
```

### Supported Publication Types

- `conference`: Conference papers
- `journal`: Journal articles
- `preprint`: Preprints (arXiv, bioRxiv, etc.)
- `workshop`: Workshop papers
- `thesis`: Thesis/dissertation
- `book`: Books and book chapters

## 🔧 Advanced Configuration

### SEO Optimization

The template includes built-in SEO optimization:

- Automatic meta tags generation
- Open Graph and Twitter Card support
- Structured data for publications
- XML sitemap generation
- RSS feed for publications

### Analytics

Add Google Analytics by including your tracking ID in `_config.yml`:

```yaml
google_analytics: "G-XXXXXXXXXX"
```

### Comments

Enable comments on research papers using Disqus:

```yaml
disqus:
  shortname: "your-disqus-shortname"
```

## 🎯 Performance

- **Lighthouse Score**: 95+ across all metrics
- **Page Load Time**: < 2 seconds
- **Mobile Friendly**: 100% mobile optimization
- **SEO Score**: 100% SEO optimization

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test locally: `bundle exec jekyll serve`
5. Commit your changes: `git commit -am 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Jekyll](https://jekyllrb.com/)
- Icons by [Feather Icons](https://feathericons.com/)
- Fonts by [Google Fonts](https://fonts.google.com/)
- Hosted on [GitHub Pages](https://pages.github.com/)

## 📞 Support

If you have any questions or need help with setup:

1. Check the [Issues](https://github.com/yourusername/academic-portfolio/issues) page
2. Create a new issue with your question
3. Provide as much detail as possible about your setup and the problem

## 🔄 Updates

To update your site with the latest template improvements:

1. Add the original template as a remote:
   ```bash
   git remote add template https://github.com/original/academic-portfolio.git
   ```

2. Fetch and merge updates:
   ```bash
   git fetch template
   git merge template/main
   ```

3. Resolve any conflicts and test locally

---

**Happy researching! 🎓**