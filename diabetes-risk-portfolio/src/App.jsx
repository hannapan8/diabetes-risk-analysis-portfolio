import React from "react";
import "./index.css";

const HERO_LINKS = [
  {
    label: "View Project on GitHub",
    href: "https://github.com/hannapan8/diabetes-risk-analysis",
    variant: "primary",
  },
  {
    label: "Open Analysis Notebook",
    href: "./notebooks/eda.ipynb",
    variant: "secondary",
  },
  {
    label: "Read Full Report",
    href: "./report.pdf",
    variant: "secondary",
  },
];

const FIGURES = [
  {
    src: "./figures/diabetes_age.png",
    alt: "Bar chart of diabetes prevalence by age group",
    caption:
      "Diabetes prevalence by age group. Risk increases as one ages.",
  },
  {
    src: "./figures/diabetes_bmi.png",
    alt: "Box plot of diabetes prevalence by BMI category",
    caption:
      "Diabetes prevalence by BMI category. People with diabetes report to have slightly higher BMIs.",
  },
  {
    src: "./figures/diabetes_education.png",
    alt: "Bar chart of diabetes prevalence by education level",
    caption:
      "There seems to be a pattern as diabetes prevalence decreases as college education level increases.",
  },
  {
    src: "./figures/diabetes_healthcare.png",
    alt: "Bar chart of diabetes prevalence by whether or not someone has healthcare",
    caption:
      "People who have healthcare are more likely to be diagnosed.",
  },
  {
    src: "./figures/diabetes_income.png",
    alt: "Bar chart of diabetes prevalence by income levels",
    caption:
      "Those with higher income are less likely to develop diabetes and vice versa.",
  },
  {
    src: "./figures/diabetes_lifestyle.png",
    alt: "Bar chart of diabetes prevalence by lifestyle factor",
    caption:
      "Lifestyle factors like physical activities and eating/drinking healthy have an effect on whether or not someone develops diabetes.",
  },
  {
    src: "./figures/diabetes_sex.png",
    alt: "Bar chart of diabetes prevalence by an individual's sex (male or female)",
    caption:
      "Men appear to be more likely to develop diabetes in this study.",
  },
];

function Pill({ children }) {
  return <span className="pill">{children}</span>;
}

function HeroLink({ href, label, variant }) {
  return (
    <a
      className={`hero-link ${variant === "secondary" ? "secondary" : ""}`}
      href={href}
      target="_blank"
      rel="noreferrer"
    >
      {label}
    </a>
  );
}

function FigureCard({ src, alt, caption }) {
  return (
    <figure className="figure-card">
      <div className="figure-media">
        <img src={src} alt={alt} loading="lazy" />
      </div>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

export default function App() {
  return (
    <div className="app-root">
      <div className="background-orbit orbit-1" />
      <div className="background-orbit orbit-2" />
      <main className="page">
        <header className="hero">
          <div className="hero-text">
            <p className="eyebrow">Disease Analysis</p>
            <h1>Predicting Diabetes Risk from Lifestyle &amp; Demographics</h1>
            <p className="hero-subtitle">
              A data visualization and modeling project using the CDC Diabetes
              Health Indicators dataset (~200k records) to explain which factors are most strongly associated with
              diabetes risk. Tech stack: Python, React, JavaScript, Jupyter.
            </p>
            <div className="pill-row">
              <Pill>Data Visualization</Pill>
              <Pill>Python · pandas · scikit-learn</Pill>
              <Pill>matplotlib · seaborn</Pill>
              <Pill>Front-end Visualization</Pill>
            </div>
            <div className="hero-links">
              {HERO_LINKS.map((link) => (
                <HeroLink key={link.label} {...link} />
              ))}
            </div>
          </div>
        </header>

        <section className="grid grid-two">
          <article className="card">
            <h2>Summary of Research Questions and Results</h2>
            <p>
              <strong>1. What lifestyle factors (physical activity, smoking, drinking, etc.) have the biggest influence in developing diabetes?{" "}</strong>
            </p>
            <p>
              Smoking and physical activity showed that they had the strongest influence on diabetes diagnoses. Smoking increased the odds of someone developing the disease by ~41% and physical activity accumulated over 50% of the feature importance, when used to classify diabetic individuals.
            </p>
            <p>
              <strong>2. Do socioeconomic factors correlate to an increased risk of developing diabetes?{" "}</strong>
            </p>
            <p>
              Yes, there are correlations between socioeconomic factors and risks of developing diabetes. Those with healthcare are significantly more likely to be diagnosed due to medical access, while income showed an inversely proportional relationship with diabetes risk.
            </p>
            <p>
              <strong>3. Are there any demographic factors that are associated with the likelihood of being diagnosed with diabetes?{" "}</strong>
            </p>
            <p>
              Yes, demographic factors have some effects on diabetes diagnoses. Age and BMI in particular, showed a positive correlation to developing diabetes. 
            </p>
          </article>

          <article className="card">
            <h2>How I Told the Story</h2>
            <p>
              I structured the project as a narrative that moves from raw data
              to an interface-ready summary:
            </p>
            <ul>
              <li>
                Exploratory charts that reveal who is most at risk in different
                subgroups.
              </li>
              <li>
                Model-based visualizations (coefficients, feature importances)
                to show which variables matter most.
              </li>
              <li>
                A layout that can be turned into a React dashboard to surface
                key metrics and charts in a product context.
              </li>
            </ul>
            <p>
              This page is a compact view of that pipeline:{" "}
              <strong>data → visuals → interpretable insights</strong>.
            </p>
          </article>
        </section>

        <section className="card gallery-card">
          <div className="section-header">
            <h2>Selected Visualizations</h2>
            <p>
              A sample of the visuals I created to explain patterns in the
              dataset. Each figure is paired with interpretation to emphasize
              storytelling, not just plotting.
            </p>
          </div>
          <div className="gallery-grid">
            {FIGURES.map((fig) => (
              <FigureCard key={fig.src} {...fig} />
            ))}
          </div>
        </section>

        <section className="card summary-card">
          <h2>Summary</h2>
          <p>
            This project combines{" "}
            <strong>data cleaning</strong>,{" "}
            <strong>visual exploration</strong>, and{" "}
            <strong>interpretable modeling</strong> to make chronic disease risk
            easier to understand. The same patterns and layouts could be
            embedded into a product-grade dashboard to support public health or
            clinical decision-making.
          </p>
          <p className="signature">
            Built by <strong>Hanna Pan</strong> · Informatics &amp; ACMS (Data
            Science &amp; Statistics), University of Washington.
          </p>
        </section>
      </main>
    </div>
  );
}
