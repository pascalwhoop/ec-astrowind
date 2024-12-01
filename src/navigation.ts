import { getPermalink, getBlogPermalink, getAsset } from './utils/permalinks';

export const headerData = {
  links: [
    {
      text: 'Home',
      href: 'https://everycure.org',
    },
    {
      text: 'About',
      href: 'https://everycure.org/about/',
    },
    {
      text: 'Blog',
      href: getBlogPermalink(),
    },
  ],
  actions: [],
};

export const footerData = {
  links: [
    {
      title: 'Blog',
      links: [
        { text: 'All Posts', href: getBlogPermalink() },
        { text: 'Categories', href: getPermalink('/category') },
        { text: 'Tags', href: getPermalink('/tag') },
      ],
    },
    {
      title: 'Company',
      links: [
        { text: 'Home', href: 'https://everycure.org' },
        { text: 'About', href: 'https://everycure.org/about/' },
      ],
    },
  ],
  secondaryLinks: [
    { text: 'Terms', href: getPermalink('/terms') },
    { text: 'Privacy Policy', href: getPermalink('/privacy') },
  ],
  socialLinks: [
    { ariaLabel: 'LinkedIn', icon: 'tabler:brand-linkedin', href: 'https://www.linkedin.com/company/everycure/' },
    { ariaLabel: 'Github', icon: 'tabler:brand-github', href: 'https://github.com/everycure' },
    { ariaLabel: 'RSS', icon: 'tabler:rss', href: getAsset('/rss.xml') },
  ],
  footNote: `
    Made with ❤️ by <a class="text-blue-600 underline dark:text-gray-200" href="https://everycure.org">Every Cure</a> · All rights reserved.
  `,
};

