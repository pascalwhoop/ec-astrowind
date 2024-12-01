---
publishDate: 2024-12-01T00:00:00Z
title: Building the Every Cure Tech Blog with AI Assistance
draft: false
excerpt: How we leveraged Cursor AI agents and the AstroWind template to quickly build our tech blog
image: ./blog-creation-cursor.jpeg
category: Technology
tags:
  - technology
  - web development
  - AI
  - astro
---

At Every Cure, we believe in leveraging technology not just for drug repurposing but also for improving our development workflow. Today, I want to share how we built this tech blog using modern tools and AI assistance, specifically Cursor AI agents and the AstroWind template.

## Starting with AstroWind

We chose [AstroWind](https://github.com/onwidget/astrowind) as our starting template because it offers:
- A modern, clean design built with Astro and Tailwind CSS
- Excellent performance out of the box
- SEO optimization
- Dark mode support
- Markdown/MDX support for blog posts

## Leveraging Cursor AI Agents

One of the most interesting aspects of building this blog was using [Cursor](https://cursor.sh/), an AI-powered code editor. Cursor's AI agents helped us:

1. **Quick Template Adaptation**: Instead of manually removing unnecessary components and pages, we used AI agents to identify and remove unused code while maintaining the site's integrity.

2. **Navigation Updates**: The AI helped us modify the navigation structure to integrate with our main website (everycure.org) while keeping the blog functionality intact.

3. **Content Focus**: We used AI to help restructure the homepage to focus on tech blog content, removing marketing sections and adding relevant tech-focused features.

Here's an example of how we structured our tech-focused features:

```typescript
items={[
  {
    title: 'Drug Repurposing Technology',
    description: 'Deep dives into our technological approach to identifying new uses for existing drugs.',
    icon: 'tabler:flask',
  },
  {
    title: 'Data Science & ML',
    description: 'Exploring how we use data science and machine learning to analyze medical data and identify potential treatments.',
    icon: 'tabler:brain',
  },
  // ...
]}
```

## The Adaptation Process

Our adaptation process was straightforward:

1. **Template Selection**: We started with the AstroWind template as our foundation.

2. **Scope Definition**: We decided to use the template initially just for our tech blog, with plans to potentially expand it to our main website later.

3. **Navigation Simplification**: We modified the navigation to:
   - Link "Home" and "About" to our main website
   - Remove unnecessary sections like "Examples" and "Download"
   - Keep the blog functionality front and center

4. **Content Cleanup**: We removed:
   - Example home pages
   - Landing pages
   - Marketing-focused sections
   - Unnecessary service and pricing pages

5. **Tech Focus**: We updated the homepage to highlight:
   - Our technical blog posts
   - Key technology areas we write about
   - Engineering insights and challenges

## Benefits of Our Approach

Using this combination of tools and approaches brought several benefits:

1. **Speed**: What could have taken days of manual adaptation was completed in hours with AI assistance.

2. **Consistency**: The AI helped ensure consistent changes across multiple files.

3. **Code Quality**: The template's high-quality code base was preserved while removing unnecessary components.

4. **Future-Proofing**: The modular structure allows us to easily expand the site's functionality in the future.

## Looking Forward

This tech blog is just the beginning. We plan to:
- Share more technical insights about our drug repurposing technology
- Discuss our engineering challenges and solutions
- Provide deep dives into our data science and ML approaches

The combination of a solid template like AstroWind and AI assistance through Cursor made it possible to quickly set up a platform where we can share these insights with the tech community.

## Conclusion

Building this tech blog demonstrated how modern development tools and AI assistance can significantly speed up web development while maintaining high quality. The combination of AstroWind's solid foundation and Cursor's AI agents allowed us to focus on what matters most: creating a platform to share our technical insights and contribute to the tech community.

Stay tuned for more technical posts about our work at Every Cure!