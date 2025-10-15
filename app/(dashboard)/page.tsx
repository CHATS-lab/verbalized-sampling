"use client";

import React from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Github, BookOpen, FileText, NotebookPen } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu';

export default function HomePage() {
  return (
    <main>
      <section className="py-10">
        <div className="max-w-6xl mx-auto px-8 sm:px-12 lg:px-16">
          {/* Paper Title and Authors Section */}
          <div className="text-center">
            <h1 className="text-4xl sm:text-4xl font-bold text-gray-700 tracking-tight mb-6">
              Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity
            </h1>

            <div className="text-lg text-gray-600 mb-2 max-w-6xl mx-auto leading-relaxed">
              <div className="mb-1">
                <a href="https://jiayizx.github.io/" className="text-blue-400 hover:text-blue-400 transition-colors">Jiayi Zhang</a><sup className="text-orange-500">1</sup><sup className="text-orange-500">†</sup>,{' '}
                <a href="https://simonucl.github.io/" className="text-blue-400 hover:text-blue-400 transition-colors">Simon Yu</a><sup className="text-orange-500">1</sup><sup className="text-orange-500">†</sup>,{' '}
                <a href="https://www.linkedin.com/in/derekch" className="text-blue-400 hover:text-blue-400 transition-colors">Derek Chong</a><sup className="text-orange-500">2</sup><sup className="text-orange-500">†</sup>,{' '}
                <a href="https://anthonysicilia.tech/" className="text-blue-400 hover:text-blue-400 transition-colors">Anthony Sicilia</a><sup className="text-orange-500">3</sup>
              </div>
              <div>
                <a href="https://tomz.people.stanford.edu/" className="text-blue-400 hover:text-blue-400 transition-colors">Michael R. Tomz</a><sup className="text-orange-500">2</sup>,{' '}
                <a href="https://nlp.stanford.edu/~manning/" className="text-blue-400 hover:text-blue-400 transition-colors">Christopher D. Manning</a><sup className="text-orange-500">2</sup>,{' '}
                <a href="https://wyshi.github.io/" className="text-blue-400 hover:text-blue-400 transition-colors">Weiyan Shi</a><sup className="text-orange-500">1</sup>
              </div>
            </div>
            
            <div className="text-lg text-black-500 mb-4">
              <sup className="text-orange-500">1</sup>Northeastern University, <sup className="text-orange-500">2</sup>Stanford University, <sup className="text-orange-500">3</sup>West Virginia University<br/>
              <sup className="text-orange-500">†</sup>Equal contribution
            </div>
            
            {/* Action Buttons */}
            <div className="flex flex-wrap justify-center gap-4 mt-10">
              <Button asChild className="rounded-full px-14 py-8 text-2xl">
                <Link href="https://arxiv.org/abs/2510.01171" className="flex items-center gap-2">
                  <FileText size={28} />
                  Paper
                </Link>
              </Button>
              <Button asChild className="rounded-full px-14 py-8 text-2xl">
                <Link href="https://simonucl.notion.site/verbalized-sampling" className="flex items-center gap-2" target="_blank" rel="noopener noreferrer">
                  <BookOpen size={28} />
                  Blog
                </Link>
              </Button>
              <Button asChild className="rounded-full px-14 py-8 text-2xl">
                <Link href="https://github.com/CHATS-lab/verbalized-sampling" className="flex items-center gap-2">
                  <Github size={28} />
                  Github
                </Link>
              </Button>
              <Button asChild className="rounded-full px-14 py-8 text-2xl">
                <Link href="https://x.com/YOUR_X_THREAD_URL" className="flex items-center gap-2">
                  <svg width="28" height="28" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                  </svg>
                  Thread
                </Link>
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button className="rounded-full flex items-center gap-2 px-14 py-8 text-2xl">
                    <NotebookPen size={28} />
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
            </div>
          </div>
        </div>
      </section>

      {/* Demo Video */}
      <section className="py-12 bg-white">
        <div className="max-w-4xl mx-auto px-8 sm:px-12 lg:px-16">
          <div className="text-center">
            <div className="text-center mb-8">
              <video 
                controls 
                className="w-full max-w-4xl mx-auto rounded-lg shadow-lg"
                poster="/images/intro_teaser.jpg"
                onEnded={(e) => {
                  const video = e.target as HTMLVideoElement;
                  video.poster = "/images/intro_teaser.jpg";
                  video.load();
                }}
              >
                <source src="/video/Demo.mp4" type="video/mp4" />
                Your browser does not support the video tag.
              </video>
              <div className="text-sm text-gray-500 mt-2">
                <strong>Figure 1 &amp; Demo:</strong> Overview of Verbalized Sampling (VS) for unlocking LLM diversity. Demo video by Qihui Fan.
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Pip Install Section */}
      <section className="py-12 bg-gray-50">
          <div className="max-w-6xl mx-auto px-8 sm:px-12 lg:px-16">
            <div className="lg:grid lg:grid-cols-2 lg:gap-12 lg:items-center">
              <div>
                  <h2 className="text-xl font-bold text-gray-700 tracking-tight sm:text-3xl mb-6">
                    Make Your LLM Output More Diverse <span className="text-orange-500">With Verbalized Sampling</span>
                  </h2>
                 <div className="space-y-4 text-base text-gray-600">
                    <p>
                      Run Verbalized Sampling and unlock diverse LLM generations in seconds. 
                      Just install and use our open-source package!
                    </p>
                    <p>
                      Check our{' '}
                      <a
                        href="https://github.com/CHATS-lab/verbalize-sampling"
                        className="underline hover:text-orange-500 font-semibold"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        GitHub
                      </a>{' '}
                      for more details.
                    </p>
                 </div>
              </div>
              <div className="mt-8 lg:mt-0">
                <div className="w-full">
                  {(() => {
                    const { Terminal } = require("./terminal_package");
                    return <Terminal height="250px"/>;
                  })()}
                </div>
              </div>
            </div>
          </div>
        </section>

      {/* BibTeX Citation */}
      <section className="py-12 bg-white">
        <div className="max-w-6xl mx-auto px-8 sm:px-12 lg:px-16">
          <div className="text-center">
            <h2 className="text-xl font-bold text-gray-700 sm:text-3xl mb-8">
              📌 BibTeX Citation
            </h2>
            <p className="text-base text-gray-600 mb-8">
              If you find our project useful, please consider citing:
            </p>
            <div className="bg-gray-50 rounded-lg shadow-sm border border-gray-200 p-6 w-full max-w-6xl mx-auto">
              <pre className="text-sm text-gray-800 font-mono whitespace-pre-wrap text-left overflow-x-auto">
{`@misc{zhang2025verbalizedsamplingmitigatemode,
      title={Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity}, 
      author={Jiayi Zhang and Simon Yu and Derek Chong and Anthony Sicilia and Michael R. Tomz and Christopher D. Manning and Weiyan Shi},
      year={2025},
      eprint={2510.01171},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.01171}, 
}`}
              </pre>
            </div>
          </div>
        </div>
      </section>

    </main>
  );
}
