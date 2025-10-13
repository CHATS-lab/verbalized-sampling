import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity | Blog',
  description: 'Blog showcasing Verbalized Sampling—a training-free method that restores diversity in aligned LLMs by 1.6-2.1× by asking for probability distributions instead of single answers. Explore results, examples, and try it yourself.',
  openGraph: {
    title: 'Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity | Blog',
    description: 'Blog showcasing Verbalized Sampling—a training-free method that restores diversity in aligned LLMs by 1.6-2.1× by asking for probability distributions instead of single answers. Explore results, examples, and try it yourself.',
    url: 'https://www.verbalized-sampling.com/blog/',
    type: 'website',
    images: [
      {
        url: 'https://www.verbalized-sampling.com/og-qualitative.jpg',
        width: 1200,
        height: 630,
        alt: 'Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity | Blog'
      }
    ]
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity | Blog',
    description: 'Blog showcasing Verbalized Sampling—a training-free method that restores diversity in aligned LLMs by 1.6-2.1× by asking for probability distributions instead of single answers. Explore results, examples, and try it yourself.',
    images: ['https://www.verbalized-sampling.com/og-qualitative.jpg']
  }
};

export default function BlogLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
