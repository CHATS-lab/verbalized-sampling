'use client';

import { Github, BookOpen, FileText, NotebookPen } from "lucide-react";

import Link from 'next/link';
import { use, useState, Suspense } from 'react';
import { Button } from '@/components/ui/button';
import { CircleIcon, Home, LogOut } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { useRouter } from 'next/navigation';
import { User } from '@/lib/db/schema';
import useSWR from 'swr';

const fetcher = (url: string) => fetch(url).then((res) => res.json());

function UserMenu() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const router = useRouter();

  async function handleSignOut() {
    router.refresh();
    router.push('/');
  }

  return (
    <>
      <Link
        href="/"
        className="text-sm font-medium text-gray-700 hover:text-gray-900"
      >
      </Link>
      <Button asChild className="rounded-full">
        <Link href="https://arxiv.org/abs/2510.01171" className="flex items-center gap-2">
          <FileText size={16} />
          Paper
        </Link>
      </Button>
      <Button asChild className="rounded-full">
        <Link href="https://www.verbalized-sampling.blog/" className="flex items-center gap-2" target="_blank" rel="noopener noreferrer">
          <BookOpen size={16} />
          Blog
        </Link>
      </Button>
      <Button asChild className="rounded-full">
        <Link href="https://github.com/CHATS-lab/verbalized-sampling" className="flex items-center gap-2">
          <Github size={16} />
          Github
        </Link>
      </Button>
      <Button asChild className="rounded-full">
        <Link href="https://x.com/YOUR_X_THREAD_URL" className="flex items-center gap-2">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
            <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
          </svg>
          X Thread
        </Link>
      </Button>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button className="rounded-full flex items-center gap-2">
            <NotebookPen size={16} />
            Notebooks
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem asChild>
            <Link href="https://colab.research.google.com/drive/1UDk4W5w6gF0dQ9Tpu0sPQethEht51GXL#offline=true&sandboxMode=true" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2">
              Direct vs. Verbalized Sampling
            </Link>
          </DropdownMenuItem>
          <DropdownMenuItem asChild>
            <Link href="https://colab.research.google.com/drive/1J18VJRnrCjIb6sTivY-znb8C3JsLQCIz#offline=true&sandboxMode=true" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2">
              Image Generation with VS
            </Link>
          </DropdownMenuItem>
          <DropdownMenuItem asChild>
            <Link href="https://colab.research.google.com/drive/1eC0nIUVC1kyANxxzhNib44qmPphdWy9o#offline=true&sandboxMode=true" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2">
              Complete Framework Tutorial
            </Link>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </>
  );
}

function Header() {
  return (
    <header className="border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
        <Link href="/" className="flex items-center">
          <CircleIcon className="h-6 w-6 text-orange-500" />
          <span className="ml-2 text-xl font-semibold text-gray-900">Verbalized Sampling</span>
        </Link>
        <div className="flex items-center space-x-4">
          <Suspense fallback={<div className="h-9" />}>
            <UserMenu />
          </Suspense>
        </div>
      </div>
    </header>
  );
}

function Footer() {
  return (
    <footer className="border-t border-gray-200 bg-gray-50 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <p className="text-center text-sm text-gray-600">
          This website is adapted from the template by{' '}
          <Link
            href="https://next-saas-start.vercel.app/"
            className="text-orange-500 hover:text-orange-600 underline"
            target="_blank"
            rel="noopener noreferrer"
          >
            Vercel
          </Link>
        </p>
      </div>
    </footer>
  );
}

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <section className="flex flex-col min-h-screen">
      <Header />
      {children}
      <Footer />
    </section>
  );
}
