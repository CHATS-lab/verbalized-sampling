import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Verbalized Sampling Blog',
  description: 'Verbalized Sampling: A training-free method to restore diversity in aligned LLMs by prompting for probability distributions.',
  openGraph: {
    title: 'Verbalized Sampling Blog',
    description: 'Verbalized Sampling: A training-free method to restore diversity in aligned LLMs by prompting for probability distributions.',
    url: 'https://www.verbalized-sampling.com/blog/',
    type: 'website',
    images: [
      {
        url: 'https://www.verbalized-sampling.com/og-qualitative.jpg',
        width: 1200,
        height: 630,
        alt: 'Verbalized Sampling Blog'
      }
    ]
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Verbalized Sampling Blog',
    description: 'Verbalized Sampling: A training-free method to restore diversity in aligned LLMs by prompting for probability distributions.',
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
