'use client';

import { Github, BookOpen, FileText } from "lucide-react";

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
        <Link href="https://github.com/Wuyxin/collabllm" className="flex items-center gap-2">
          <Github size={16} />
          Github
        </Link>
      </Button>
      <Button asChild className="rounded-full">
        <Link href="#blog" className="flex items-center gap-2">
          <BookOpen size={16} />
          Blog
        </Link>
      </Button>
      <Button asChild className="rounded-full">
        <Link href="https://arxiv.org/pdf/2502.00640" className="flex items-center gap-2">
          <FileText size={16} />
          Paper
        </Link>
      </Button>
    </>
  );
}

function Header() {
  return (
    <header className="border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
        <Link href="/" className="flex items-center">
          <CircleIcon className="h-6 w-6 text-orange-500" />
          <span className="ml-2 text-xl font-semibold text-gray-900">CollabLLM</span>
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

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <section className="flex flex-col min-h-screen">
      <Header />
      {children}
    </section>
  );
}
