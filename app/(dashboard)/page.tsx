"use client";

import React from 'react';
import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Github, BookOpen, FileText, NotebookPen, Image as ImageIcon } from 'lucide-react';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu';
import { Terminal_Prompt } from './terminal_prompt';

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
            <div className="flex flex-wrap justify-center gap-4 mt-8">
              <Button asChild className="rounded-full px-8 py-4 text-lg">
                <Link href="https://arxiv.org/abs/2510.01171" className="flex items-center gap-3">
                  <FileText size={20} />
                  Paper
                </Link>
              </Button>
              <Button asChild className="rounded-full px-8 py-4 text-lg">
                <Link href="https://simonucl.notion.site/verbalized-sampling" className="flex items-center gap-3" target="_blank" rel="noopener noreferrer">
                  <BookOpen size={20} />
                  Blog
                </Link>
              </Button>
              <Button asChild className="rounded-full px-8 py-4 text-lg">
                <Link href="https://github.com/CHATS-lab/verbalized-sampling" className="flex items-center gap-3">
                  <Github size={20} />
                  Github
                </Link>
              </Button>
              <Button asChild className="rounded-full px-8 py-4 text-lg">
                <Link href="https://tinyurl.com/vs-gallery" className="flex items-center gap-3">
                  <ImageIcon size={20} />
                  Examples
                </Link>
              </Button>
              <Button asChild className="rounded-full px-8 py-4 text-lg">
                <Link href="https://x.com/shi_weiyan/status/1978453313096908916" className="flex items-center gap-3">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                  </svg>
                  X Thread
                </Link>
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button className="rounded-full flex items-center gap-3 px-8 py-4 text-lg">
                    <NotebookPen size={20} />
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
              <Button asChild className="rounded-full px-8 py-4 text-lg">
                <Link href="https://www.youtube.com/watch?v=VoBdywmdim0" className="flex items-center gap-3" target="_blank" rel="noopener noreferrer">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M10 8.64L15.27 12 10 15.36V8.64M12 2C6.5 2 2 6.5 2 12S6.5 22 12 22 22 17.5 22 12 17.5 2 12 2M12 20C7.59 20 4 16.41 4 12S7.59 4 12 4 20 7.59 20 12 16.41 20 12 20Z" />
                  </svg>
                  Podcast
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

     {/* Media Coverage Section */}
     <section className="py-6 bg-white">
        <div className="max-w-4xl mx-auto px-8 sm:px-12 lg:px-16">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-2xl sm:text-2xl font-bold text-gray-800 whitespace-nowrap">Featured On:</h2>
            {/* VentureBeat */}
            <a
              href="https://venturebeat.com/ai/researchers-find-adding-this-one-simple-sentence-to-prompts-makes-ai-models"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border-2 border-black rounded-full hover:bg-gray-50 transition-all shadow-sm"
            >
              <span className="text-red-500 font-bold text-xs">VB</span>
              <span className="font-semibold text-gray-800 text-sm">VentureBeat</span>
            </a>

            {/* Forbes */}
            <a
              href="https://www.forbes.com/sites/lanceeliot/2025/11/01/prompt-engineering-newest-technique-is-verbalized-sampling-that-stirs-ai-to-be-free-thinking-and-improve-your-responses/?ss=ai"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border-2 border-black rounded-full hover:bg-gray-50 transition-all shadow-sm"
            >
              <span className="text-blue-600 font-bold text-xs">F</span>
              <span className="font-semibold text-gray-800 text-sm">Forbes</span>
            </a>

            {/* Yahoo News */}
            <a
              href="https://currently.att.yahoo.com/att/magic-prompt-allegedly-makes-chatgpt-180523549.html"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border-2 border-black rounded-full hover:bg-gray-50 transition-all shadow-sm"
            >
              <span className="text-purple-700 font-bold text-xs">Y!</span>
              <span className="font-semibold text-gray-800 text-sm">Yahoo News</span>
            </a>

            {/* TLDR AI */}
            <a
              href="https://tldr.tech/ai/2025-10-16"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border-2 border-black rounded-full hover:bg-gray-50 transition-all shadow-sm"
            >
              <span className="text-blue-500 font-bold text-xs">📰</span>
              <span className="font-semibold text-gray-800 text-sm">TLDR AI</span>
            </a>

            {/* Medium */}
            <a
              href="https://generativeai.pub/stanford-just-killed-prompt-engineering-with-8-words-and-i-cant-believe-it-worked-8349d6524d2b"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border-2 border-black rounded-full hover:bg-gray-50 transition-all shadow-sm"
            >
              <span className="text-black font-bold text-xs">M</span>
              <span className="font-semibold text-gray-800 text-sm">Medium</span>
            </a>

            {/* Neuron Daily */}
            <a
              href="https://www.theneurondaily.com/p/new-ai-models-introducing-veo-3-1-and-claude-haiku-4-5"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border-2 border-black rounded-full hover:bg-gray-50 transition-all shadow-sm"
            >
              <span className="text-orange-500 font-bold text-xs">🧠</span>
              <span className="font-semibold text-gray-800 text-sm">Neuron Daily</span>
            </a>

            {/* Analytics Vidhya */}
            <a
              href="https://www.analyticsvidhya.com/blog/2025/10/verbalized-sampling/"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border-2 border-black rounded-full hover:bg-gray-50 transition-all shadow-sm"
            >
              <span className="text-green-600 font-bold text-xs">📊</span>
              <span className="font-semibold text-gray-800 text-sm">Analytics Vidhya</span>
            </a>

            {/* AI Papers Podcast Daily */}
            <a
              href="https://www.youtube.com/watch?v=VoBdywmdim0"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border-2 border-black rounded-full hover:bg-gray-50 transition-all shadow-sm"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="red" className="inline-block">
                <path d="M10 8.64L15.27 12 10 15.36V8.64M12 2C6.5 2 2 6.5 2 12S6.5 22 12 22 22 17.5 22 12 17.5 2 12 2M12 20C7.59 20 4 16.41 4 12S7.59 4 12 4 20 7.59 20 12 16.41 20 12 20Z" />
              </svg>
              <span className="font-semibold text-gray-800 text-sm">AI Papers Podcast</span>
            </a>

            {/* AI in Education Daily Podcast */}
            <a
              href="https://podfollow.com/ai-in-education-daily/episode/8650510757ba9be732ddcaa600ec4403d1229020/view"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border-2 border-black rounded-full hover:bg-gray-50 transition-all shadow-sm"
            >
              <span className="text-indigo-600 font-bold text-xs">🎓</span>
              <span className="font-semibold text-gray-800 text-sm">AI in Education</span>
            </a>

            {/* Daily Dose of Data Science */}
            <a
              href="https://blog.dailydoseofds.com/p/verbalized-sampling-in-llms"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-white border-2 border-black rounded-full hover:bg-gray-50 transition-all shadow-sm"
            >
              <img 
                src="https://substackcdn.com/image/fetch/w_80,h_80,c_fill,f_webp,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc5dc1fee-2d1e-4892-b219-4b96f6998ab5_288x288.png"
                alt="Daily Dose of Data Science"
                className="w-4 h-4 rounded-full"
              />
              <span className="font-semibold text-gray-800 text-sm">Daily Dose of DS</span>
            </a>
          </div>
        </div>
      </section>

      {/* decrease section 间距 */}
      <style jsx global>{`
        section.py-12 {
          padding-top: 0.2rem !important;
          padding-bottom: 0.2rem !important;
        }
      `}</style>

      <section className="py-12 bg-white">
        <div className="max-w-4xl mx-auto px-8 sm:px-12 lg:px-16">
          <div className="text-center">
            {/* Demo Video */}
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

            <h2 className="text-xl font-bold text-gray-700 tracking-tight sm:text-3xl mb-8">
              Abstract
            </h2>

            <div className="max-w-4xl mx-auto">
              <p className="text-base text-gray-600 leading-relaxed text-left">
                Post-training alignment often reduces LLM diversity, leading to a phenomenon known as <em>mode collapse</em>. 
                Unlike prior work that attributes this effect to algorithmic limitations, we identify a fundamental, pervasive data-level driver: <em>typicality bias</em> in preference data, 
                whereby annotators systematically favor familiar text as a result of well-established findings in cognitive psychology. 
                We formalize this bias theoretically, verify it empirically on preference datasets, and show that it plays a central role in mode collapse. 
              </p>
              <p className="text-base text-gray-600 leading-relaxed text-left mt-6">
                Motivated by this analysis, we introduce <strong>Verbalized Sampling (VS)</strong>, a simple, training-free prompting method to circumvent mode collapse. VS prompts the model to verbalize a probability distribution over a set of responses (e.g., "Generate 5 jokes about coffee and their corresponding probabilities").
                Comprehensive experiments show that VS significantly improves performance across creative writing (poems, stories, jokes), dialogue simulation, open-ended QA, and synthetic data generation, without sacrificing factual accuracy and safety. For instance, in creative writing, VS increases diversity by 1.6-2.1× over direct prompting. We further observe an emergent trend that more capable models benefit more from VS.
                In sum, our work provides a new data-centric perspective on mode collapse and a practical inference-time remedy that helps unlock pre-trained generative diversity.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Pip Install Section */}
      {/* <section className="py-12 bg-gray-50">
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
        </section> */}

      {/* Terminal Prompt (after Pip Install) */}
      <section className="py-16 bg-white w-full">
        <div className="max-w-6xl mx-auto px-8 sm:px-12 lg:px-16">
          <div className="lg:grid lg:grid-cols-12 lg:gap-4 lg:items-start">
            <div className="lg:col-span-7 text-left sm:text-center lg:text-left">
              <h2 className="text-xl font-bold text-gray-700 tracking-tight sm:text-3xl">
                Try It Yourself: <span className="text-orange-500">The Magic Prompt</span>
              </h2>
              <p className="text-base text-gray-600 mt-4">
                Use this prompt to sample multiple responses with explicit probabilities with your favorite LLM. 
                Copy it into your provider's playground, API call, or chat interface, then replace "Tell me a short story about a bear" with your task.
                <br />
                <br />
                For best results, we recommend starting with models like GPT-5, Claude Opus 4, and Gemini 2.5 Pro.
                <br />
                <br />
                Please refer to our&nbsp;
                <a
                  href="https://github.com/CHATS-lab/verbalize-sampling#prompt-templates"
                  className="underline hover:text-orange-500 font-semibold"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  GitHub
                </a>
                &nbsp;for additional prompt variations and examples. And check out this <a href="https://x.com/dch/status/1978471395173740900" className="underline hover:text-orange-500 font-semibold" target="_blank" rel="noopener noreferrer">X thread</a> for more practical tips and troubleshooting help!
              </p>
            </div>
            <div className="mt-8 lg:mt-0 lg:col-span-5 lg:mx-0 w-full">
              <div className="w-full">
                <Terminal_Prompt />
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
