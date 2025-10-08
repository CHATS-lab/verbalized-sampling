"use client";

import React, { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { ArrowRight, CreditCard, Database, Calendar, User, Clock, Eye, Heart, ThumbsUp, Share2 } from 'lucide-react';

import { Terminal } from './terminal';

// Type definitions
interface VisitorSession {
  visitorId: string;
  firstVisit: number;
  lastActivity: number;
}

interface Reactions {
  heart: number;
  thumbsUp: number;
  share: number;
}

type ReactionType = keyof Reactions;

type SharePlatform = 'twitter' | 'linkedin' | 'slack' | 'email' | 'copy';

interface ShareUrls {
  twitter: string;
  linkedin: string;
  slack: string;
  email: string;
}

interface ReactionButton {
  type: ReactionType;
  icon: React.ComponentType<{ size?: number; className?: string }>;
  color: string;
  label: string;
}

// Reset all CollabLLM data
// localStorage.removeItem('collabllm_view_count');
// localStorage.removeItem('collabllm_reactions');
// localStorage.removeItem('collabllm_user_reactions');
// localStorage.removeItem('collabllm_visitor_sessions');
// localStorage.removeItem('collabllm_last_visit');
// console.log('✅ All CollabLLM data reset!');
// location.reload(); // Refresh the page


// Viewer System Component with Persistent Storage
const ViewerSystem = () => {
  const [viewCount, setViewCount] = useState<number>(0);
  const [reactions, setReactions] = useState<Reactions>({
    heart: 0,
    thumbsUp: 0,
    share: 0
  });
  const [userReactions, setUserReactions] = useState<Set<string>>(new Set());
  const [showReactionAnimation, setShowReactionAnimation] = useState<string | null>(null);
  const [showShareModal, setShowShareModal] = useState<boolean>(false);
  const [isLocalStorageAvailable, setIsLocalStorageAvailable] = useState<boolean>(true);
  
  // Generate or retrieve persistent visitor ID
  const [uniqueVisitorId] = useState<string>(() => {
    try {
      let visitorId = localStorage.getItem('collabllm_visitor_id');
      if (!visitorId) {
        visitorId = 'visitor_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
        localStorage.setItem('collabllm_visitor_id', visitorId);
      }
      return visitorId;
    } catch {
      return 'visitor_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }
  });

  // Storage keys
  const STORAGE_KEYS = {
    viewCount: 'collabllm_view_count',
    reactions: 'collabllm_reactions',
    userReactions: 'collabllm_user_reactions',
    lastVisit: 'collabllm_last_visit',
    visitorSessions: 'collabllm_visitor_sessions',
    visitorId: 'collabllm_visitor_id'
  };

  // Helper function to safely use localStorage
  const safeLocalStorage = {
    getItem: (key: string): string | null => {
      try {
        return localStorage.getItem(key);
      } catch {
        return null;
      }
    },
    setItem: (key: string, value: string): boolean => {
      try {
        localStorage.setItem(key, value);
        return true;
      } catch {
        return false;
      }
    },
    removeItem: (key: string): boolean => {
      try {
        localStorage.removeItem(key);
        return true;
      } catch {
        return false;
      }
    }
  };

  // Initialize data from localStorage and track visitor
  useEffect(() => {
    // Check localStorage availability
    try {
      const testKey = '__localStorage_test__';
      localStorage.setItem(testKey, 'test');
      localStorage.removeItem(testKey);
      setIsLocalStorageAvailable(true);
    } catch {
      setIsLocalStorageAvailable(false);
      console.warn('LocalStorage not available, using memory storage');
    }

    if (!isLocalStorageAvailable) {
      // Set default values for non-localStorage environment
      setViewCount(1);
      setReactions({ heart: 0, thumbsUp: 0, share: 0 });
      return;
    }

    try {
      // Load existing data or set defaults
      const savedViewCount = safeLocalStorage.getItem(STORAGE_KEYS.viewCount);
      const savedReactions = safeLocalStorage.getItem(STORAGE_KEYS.reactions);
      const savedUserReactions = safeLocalStorage.getItem(STORAGE_KEYS.userReactions);
      const savedVisitorSessions = safeLocalStorage.getItem(STORAGE_KEYS.visitorSessions);
      
      // Set reactions from storage or start at 0
      if (savedReactions) {
        const loadedReactions = JSON.parse(savedReactions);
        // Ensure all reaction types exist, even if they weren't in the saved data
        setReactions({
          heart: loadedReactions.heart || 0,
          thumbsUp: loadedReactions.thumbsUp || 0,
          share: loadedReactions.share || 0
        });
      } else {
        const initialReactions = { heart: 0, thumbsUp: 0, share: 0 };
        setReactions(initialReactions);
        safeLocalStorage.setItem(STORAGE_KEYS.reactions, JSON.stringify(initialReactions));
      }
      
      if (savedUserReactions) {
        const userReactionArray = JSON.parse(savedUserReactions);
        setUserReactions(new Set(userReactionArray));
      }

      // Track unique visitors with improved logic
      const visitorSessions: VisitorSession[] = savedVisitorSessions ? JSON.parse(savedVisitorSessions) : [];
      const now = Date.now();
      const sessionTimeout = 30 * 60 * 1000; // 30 minutes
      
      // Clean old sessions
      const activeSessions = visitorSessions.filter((session: VisitorSession) => 
        now - session.lastActivity < sessionTimeout
      );
      
      // Check if this visitor has an active session
      const existingSession = activeSessions.find((session: VisitorSession) => 
        session.visitorId === uniqueVisitorId
      );
      
      if (!existingSession) {
        // New visitor or expired session - increment view count
        activeSessions.push({
          visitorId: uniqueVisitorId,
          firstVisit: now,
          lastActivity: now
        });
        
        const currentCount = savedViewCount ? parseInt(savedViewCount) : 0;
        const newViewCount = currentCount + 1;
        setViewCount(newViewCount);
        safeLocalStorage.setItem(STORAGE_KEYS.viewCount, newViewCount.toString());
      } else {
        // Existing visitor - just update activity and load current view count
        existingSession.lastActivity = now;
        setViewCount(savedViewCount ? parseInt(savedViewCount) : 1);
      }
      
      // Save updated sessions
      safeLocalStorage.setItem(STORAGE_KEYS.visitorSessions, JSON.stringify(activeSessions));
      safeLocalStorage.setItem(STORAGE_KEYS.lastVisit, now.toString());
      
    } catch (error) {
      console.warn('Error initializing viewer data:', error);
      // Fallback to default values
      setViewCount(1);
      setReactions({ heart: 0, thumbsUp: 0, share: 0 });
    }
  }, [uniqueVisitorId, isLocalStorageAvailable]);

  // Periodic updates - simulate real-time activity (reduced frequency and better logic)
  useEffect(() => {
    if (!isLocalStorageAvailable) return;

    const interval = setInterval(() => {
      try {
        // Only simulate occasional new visitors, not frequent ones
        if (Math.random() < 0.05) { // 5% chance every 30 seconds (reduced from 20% every 10 seconds)
          const savedViewCount = safeLocalStorage.getItem(STORAGE_KEYS.viewCount);
          const currentCount = savedViewCount ? parseInt(savedViewCount) : 0;
          const increment = Math.floor(Math.random() * 2) + 1; // 1-2 new views
          const newViewCount = currentCount + increment;
          
          setViewCount(newViewCount);
          safeLocalStorage.setItem(STORAGE_KEYS.viewCount, newViewCount.toString());
          
          // Add simulated visitor sessions
          const savedVisitorSessions = safeLocalStorage.getItem(STORAGE_KEYS.visitorSessions);
          const visitorSessions: VisitorSession[] = savedVisitorSessions ? JSON.parse(savedVisitorSessions) : [];
          const now = Date.now();
          const sessionTimeout = 30 * 60 * 1000;
          
          // Clean old sessions first
          const activeSessions = visitorSessions.filter((session: VisitorSession) => 
            now - session.lastActivity < sessionTimeout
          );
          
          // Add new simulated sessions
          for (let i = 0; i < increment; i++) {
            activeSessions.push({
              visitorId: 'simulated_' + Math.random().toString(36).substr(2, 9) + '_' + now,
              firstVisit: now,
              lastActivity: now
            });
          }
          
          safeLocalStorage.setItem(STORAGE_KEYS.visitorSessions, JSON.stringify(activeSessions));
        }
      } catch (error) {
        console.warn('Error updating visitor data:', error);
      }
    }, 30000); // Check every 30 seconds instead of 10

    return () => clearInterval(interval);
  }, [isLocalStorageAvailable]);

  // Update last activity periodically
  useEffect(() => {
    const activityInterval = setInterval(() => {
      try {
        const savedVisitorSessions = localStorage.getItem(STORAGE_KEYS.visitorSessions);
        if (savedVisitorSessions) {
          const visitorSessions: VisitorSession[] = JSON.parse(savedVisitorSessions);
          const now = Date.now();
          
          const updatedSessions = visitorSessions.map((session: VisitorSession) => 
            session.visitorId === uniqueVisitorId 
              ? { ...session, lastActivity: now }
              : session
          );
          
          localStorage.setItem(STORAGE_KEYS.visitorSessions, JSON.stringify(updatedSessions));
        }
      } catch (error) {
        console.warn('Error updating activity:', error);
      }
    }, 60000); // Update every minute

    return () => clearInterval(activityInterval);
  }, [uniqueVisitorId]);

  const handleReaction = (reactionType: string) => {
    // Special handling for share - always increment and show social media options
    if (reactionType === 'share') {
      handleShare();
      return;
    }

    const newUserReactions = new Set(userReactions);
    
    try {
      if (userReactions.has(reactionType)) {
        // Remove reaction
        newUserReactions.delete(reactionType);
        setReactions(prev => {
          const updated = {
            ...prev,
            [reactionType as keyof Reactions]: Math.max(0, prev[reactionType as keyof Reactions] - 1)
          };
          localStorage.setItem(STORAGE_KEYS.reactions, JSON.stringify(updated));
          return updated;
        });
      } else {
        // Add reaction
        newUserReactions.add(reactionType);
        setReactions(prev => {
          const updated = {
            ...prev,
            [reactionType as keyof Reactions]: prev[reactionType as keyof Reactions] + 1
          };
          localStorage.setItem(STORAGE_KEYS.reactions, JSON.stringify(updated));
          return updated;
        });
        
        // Show animation
        setShowReactionAnimation(reactionType);
        setTimeout(() => setShowReactionAnimation(null), 500);
      }
      
      setUserReactions(newUserReactions);
      localStorage.setItem(STORAGE_KEYS.userReactions, JSON.stringify([...newUserReactions]));
      
    } catch (error) {
      console.warn('Error saving reaction data:', error);
    }
  };

  const handleShare = () => {
    // Increment share count
    setReactions(prev => {
      const updated = {
        ...prev,
        share: prev.share + 1
      };
      try {
        localStorage.setItem(STORAGE_KEYS.reactions, JSON.stringify(updated));
      } catch (error) {
        console.warn('Error saving share data:', error);
      }
      return updated;
    });

    // Show animation
    setShowReactionAnimation('share');
    setTimeout(() => setShowReactionAnimation(null), 500);

    // Create share data
    const shareData = {
      title: 'CollabLLM: From Passive Responders to Active Collaborators',
      text: 'Sharing the blog: "Building the Future of Collaborative AI: Our Journey with CollabLLM" - A unified fine-tuning framework that optimizes LLMs for effective multiturn collaboration.',
      url: `${window.location.origin}${window.location.pathname}#blog`
    };

    // Try native share API first (mobile devices)
    if (navigator.share && navigator.canShare && navigator.canShare(shareData)) {
      navigator.share(shareData).catch(() => {
        // Fallback to modal if native share fails
        setShowShareModal(true);
      });
    } else {
      // Desktop fallback - show share modal
      setShowShareModal(true);
    }
  };

  const shareToSocial = (platform: SharePlatform) => {
    const shareData = {
      title: 'CollabLLM: From Passive Responders to Active Collaborators',
      text: 'Sharing the blog: "Building the Future of Collaborative AI: Our Journey with CollabLLM" - A unified fine-tuning framework that optimizes LLMs for effective multiturn collaboration.',
      url: `${window.location.origin}${window.location.pathname}#blog`
    };

    const shareUrls: ShareUrls = {
      twitter: `https://twitter.com/messages/compose?text=${encodeURIComponent(`${shareData.text} ${shareData.url}`)}`,
      linkedin: `https://www.linkedin.com/messaging/`,
      slack: `slack://open`,
      email: `mailto:?subject=${encodeURIComponent(shareData.title)}&body=${encodeURIComponent(`${shareData.text}\n\n${shareData.url}`)}`
    };

    if (platform === 'copy') {
      // Copy link to clipboard
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(shareData.url).then(() => {
          alert('✅ Link copied to clipboard!');
          setShowShareModal(false);
        }).catch(() => {
          // Fallback for clipboard API failure
          const textArea = document.createElement('textarea');
          textArea.value = shareData.url;
          document.body.appendChild(textArea);
          textArea.select();
          try {
            document.execCommand('copy');
            alert('✅ Link copied to clipboard!');
            setShowShareModal(false);
          } catch (err) {
            prompt('Copy this link manually:', shareData.url);
          }
          document.body.removeChild(textArea);
        });
      } else {
        // Old browser fallback
        prompt('Copy this link:', shareData.url);
        setShowShareModal(false);
      }
    } else if (platform === 'linkedin') {
      // Special handling for LinkedIn DM - copy message and open LinkedIn
      const linkedinMessage = `${shareData.text}\n\n${shareData.url}`;
      
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(linkedinMessage).then(() => {
          alert('✅ Message copied to clipboard! Opening LinkedIn...\n\nPaste the message in your DM.');
          window.open(shareUrls.linkedin, '_blank', 'width=800,height=600,scrollbars=yes,resizable=yes');
          setShowShareModal(false);
        }).catch(() => {
          // Fallback
          prompt('Copy this message for LinkedIn DM:', linkedinMessage);
          window.open(shareUrls.linkedin, '_blank', 'width=800,height=600,scrollbars=yes,resizable=yes');
          setShowShareModal(false);
        });
      } else {
        // Old browser fallback
        prompt('Copy this message for LinkedIn DM:', linkedinMessage);
        window.open(shareUrls.linkedin, '_blank', 'width=800,height=600,scrollbars=yes,resizable=yes');
        setShowShareModal(false);
      }
    } else if (platform === 'slack') {
      // Special handling for Slack - copy message and open Slack
      const slackMessage = `${shareData.text}\n\n${shareData.url}`;
      
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(slackMessage).then(() => {
          alert('✅ Message copied to clipboard! Opening Slack...\n\nPaste the message in any channel or DM.');
          window.open(shareUrls.slack, '_blank');
          setShowShareModal(false);
        }).catch(() => {
          // Fallback
          prompt('Copy this message for Slack:', slackMessage);
          window.open(shareUrls.slack, '_blank');
          setShowShareModal(false);
        });
      } else {
        // Old browser fallback
        prompt('Copy this message for Slack:', slackMessage);
        window.open(shareUrls.slack, '_blank');
        setShowShareModal(false);
      }
    } else {
      // Open other social media share URLs
      const newWindow = window.open(shareUrls[platform as keyof ShareUrls], '_blank', 'width=600,height=400,scrollbars=yes,resizable=yes');
      if (!newWindow) {
        alert('Pop-up blocked! Please allow pop-ups for sharing or copy the link manually.');
      }
      setShowShareModal(false);
    }
  };

  // Share Modal Component
  const ShareModal = () => {
    if (!showShareModal) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50" onClick={() => setShowShareModal(false)}>
        <div className="bg-white rounded-lg p-6 max-w-sm w-full mx-4 shadow-lg" onClick={e => e.stopPropagation()}>
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Share this article</h3>
          
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => shareToSocial('twitter')}
              className="flex items-center gap-2 p-3 border border-gray-200 rounded-lg hover:bg-blue-50 hover:border-blue-300 transition-colors"
            >
              <div className="w-5 h-5 bg-blue-400 rounded"></div>
              <span className="text-sm font-medium">Twitter DM</span>
            </button>
            
            <button
              onClick={() => shareToSocial('linkedin')}
              className="flex items-center gap-2 p-3 border border-gray-200 rounded-lg hover:bg-blue-50 hover:border-blue-300 transition-colors"
            >
              <div className="w-5 h-5 bg-blue-600 rounded"></div>
              <span className="text-sm font-medium">LinkedIn DM</span>
            </button>
            
            <button
              onClick={() => shareToSocial('slack')}
              className="flex items-center gap-2 p-3 border border-gray-200 rounded-lg hover:bg-purple-50 hover:border-purple-300 transition-colors"
            >
              <div className="w-5 h-5 bg-purple-500 rounded"></div>
              <span className="text-sm font-medium">Slack</span>
            </button>
            
            <button
              onClick={() => shareToSocial('email')}
              className="flex items-center gap-2 p-3 border border-gray-200 rounded-lg hover:bg-gray-50 hover:border-gray-300 transition-colors"
            >
              <div className="w-5 h-5 bg-gray-500 rounded"></div>
              <span className="text-sm font-medium">Email</span>
            </button>
            
            <button
              onClick={() => shareToSocial('copy')}
              className="flex items-center gap-2 p-3 border border-gray-200 rounded-lg hover:bg-green-50 hover:border-green-300 transition-colors col-span-2"
            >
              <div className="w-5 h-5 bg-green-500 rounded"></div>
              <span className="text-sm font-medium">Copy Link</span>
            </button>
          </div>
          
          <button
            onClick={() => setShowShareModal(false)}
            className="w-full mt-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    );
  };

  const reactionButtons: ReactionButton[] = [
    { type: 'heart', icon: Heart, color: 'text-red-500', label: 'Love' },
    { type: 'thumbsUp', icon: ThumbsUp, color: 'text-blue-500', label: 'Good' },
    { type: 'share', icon: Share2, color: 'text-green-500', label: 'Share' }
  ];

  return (
    <>
      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 py-3">
        {/* Share Button Only */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => handleReaction('share')}
            className="flex items-center gap-2 px-4 py-2 rounded-full border border-gray-300 hover:border-gray-400 text-gray-500 hover:text-green-500 transition-all duration-200 hover:scale-105"
            title="Share this article"
          >
            <Share2 size={16} />
            <span className="text-sm font-medium">Share</span>
          </button>
        </div>
      </div>
      {/* Share Modal */}
      <ShareModal />
    </>
  );
};

export default function HomePage() {
  return (
    <main>
      <section className="py-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Paper Title and Authors Section */}
          <div className="text-center">
            <h1 className="text-4xl font-bold text-gray-700 tracking-tight sm:text-5xl mb-6">
              CollabLLM: From Passive Responders to Active Collaborators
            </h1>
            
            <div className="text-xl text-gray-600 mb-2 max-w-5xl mx-auto leading-relaxed">
              <div className="mb-1">
                <a href="https://cs.stanford.edu/~shirwu/" className="text-blue-400 hover:text-blue-400 transition-colors">Shirley Wu</a><sup className="text-orange-500">1</sup>,{' '}
                <a href="https://www.microsoft.com/en-us/research/people/mgalley/" className="text-blue-400 hover:text-blue-400 transition-colors">Michel Galley</a><sup className="text-orange-500">2</sup>,{' '}
                <a href="https://www.microsoft.com/en-us/research/people/baolinpeng/" className="text-blue-400 hover:text-blue-400 transition-colors">Baolin Peng</a><sup className="text-orange-500">2</sup>,{' '}
                <a href="https://sites.google.com/site/hcheng2site" className="text-blue-400 hover:text-blue-400 transition-colors">Hao Cheng</a><sup className="text-orange-500">2</sup>,{' '}
                <a href="https://scholar.google.com/citations?user=jJglcU8AAAAJ&hl=en" className="text-blue-400 hover:text-blue-400 transition-colors">Gavin Li</a><sup className="text-orange-500">1</sup>,{' '}
                <a href="https://yao-dou.github.io/" className="text-blue-400 hover:text-blue-400 transition-colors">Yao Dou</a><sup className="text-orange-500">3</sup>,{' '}
                <a href="https://www.linkedin.com/in/wilsoncai" className="text-blue-400 hover:text-blue-400 transition-colors">Weixin Cai</a><sup className="text-orange-500">1</sup>
              </div>
              <div>
                <a href="https://www.james-zou.com/" className="text-blue-400 hover:text-blue-400 transition-colors">James Zou</a><sup className="text-orange-500">1</sup>,{' '}
                <a href="https://cs.stanford.edu/people/jure/" className="text-blue-400 hover:text-blue-400 transition-colors">Jure Leskovec</a><sup className="text-orange-500">1</sup>,{' '}
                <a href="https://www.microsoft.com/en-us/research/people/jfgao/" className="text-blue-400 hover:text-blue-400 transition-colors">Jianfeng Gao</a><sup className="text-orange-500">2</sup>
              </div>
            </div>
            
            <div className="text-xl text-black-500 mb-4">
              <sup className="text-orange-500">1</sup>Stanford University, <sup className="text-orange-500">2</sup>Microsoft, <sup className="text-orange-500">3</sup>Georgia Tech
            </div>
            
            <div className="text-xl font-bold">
              <span className="text-orange-600">ICML 2025 Outstanding Paper</span>
            </div>
          </div>
        </div>
      </section>
      

      <section className="py-8  bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:grid lg:grid-cols-12 lg:gap-8">
            <div className="sm:text-center md:max-w-2xl md:mx-auto lg:col-span-7 lg:text-left">
              <h2 className="text-3xl font-bold text-gray-700 tracking-tight sm:text-4xl mb-4">
                Make Your LLMs
                <span className="block text-orange-500">Active Collaborators</span>
              </h2>
              <p className="mt-3 text-base text-gray-500 sm:mt-5 sm:text-xl lg:text-lg xl:text-xl">
              CollabLLM is a unified fine-tuning framework that optimizes LLMs for 
              effective and efficient multiturn collaboration with users.
              </p>
              <div className="mt-8 sm:max-w-lg sm:mx-auto sm:text-center lg:text-left lg:mx-0">
                <a
                  href="https://huggingface.co/collabllm"
                  target="_blank"
                >
                  <Button
                    size="lg"
                    variant="outline"
                    className="text-lg rounded-full"
                  >
                    Try CollabLLM Models
                    <ArrowRight className="ml-2 h-5 w-5" />
                  </Button>
                </a>
              </div>
            </div>
            <div className="mt-12 relative sm:max-w-lg sm:mx-auto lg:mt-0 lg:max-w-none lg:mx-0 lg:col-span-5 lg:flex lg:items-center">
              <Terminal />
            </div>
          </div>
        </div>
      </section>

      <section className="py-16 bg-gray-50 w-full">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:grid lg:grid-cols-3 lg:gap-8">
            <div>
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-orange-500 text-white">
                <svg viewBox="0 0 24 24" className="h-6 w-6">
                  <path
                    fill="currentColor"
                    d="M14.23 12.004a2.236 2.236 0 0 1-2.235 2.236 2.236 2.236 0 0 1-2.236-2.236 2.236 2.236 0 0 1 2.235-2.236 2.236 2.236 0 0 1 2.236 2.236zm2.648-10.69c-1.346 0-3.107.96-4.888 2.622-1.78-1.653-3.542-2.602-4.887-2.602-.41 0-.783.093-1.106.278-1.375.793-1.683 3.264-.973 6.365C1.98 8.917 0 10.42 0 12.004c0 1.59 1.99 3.097 5.043 4.03-.704 3.113-.39 5.588.988 6.38.32.187.69.275 1.102.275 1.345 0 3.107-.96 4.888-2.624 1.78 1.654 3.542 2.603 4.887 2.603.41 0 .783-.09 1.106-.275 1.374-.792 1.683-3.263.973-6.365C22.02 15.096 24 13.59 24 12.004c0-1.59-1.99-3.097-5.043-4.032.704-3.11.39-5.587-.988-6.38-.318-.184-.688-.277-1.092-.278zm-.005 1.09v.006c.225 0 .406.044.558.127.666.382.955 1.835.73 3.704-.054.46-.142.945-.25 1.44-.96-.236-2.006-.417-3.107-.534-.66-.905-1.345-1.727-2.035-2.447 1.592-1.48 3.087-2.292 4.105-2.295zm-9.77.02c1.012 0 2.514.808 4.11 2.28-.686.72-1.37 1.537-2.02 2.442-1.107.117-2.154.298-3.113.538-.112-.49-.195-.964-.254-1.42-.23-1.868.054-3.32.714-3.707.19-.09.4-.127.563-.132zm4.882 3.05c.455.468.91.992 1.36 1.564-.44-.02-.89-.034-1.345-.034-.46 0-.915.01-1.36.034.44-.572.895-1.096 1.345-1.565zM12 8.1c.74 0 1.477.034 2.202.093.406.582.802 1.203 1.183 1.86.372.64.71 1.29 1.018 1.946-.308.655-.646 1.31-1.013 1.95-.38.66-.773 1.288-1.18 1.87-.728.063-1.466.098-2.21.098-.74 0-1.477-.035-2.202-.093-.406-.582-.802-1.204-1.183-1.86-.372-.64-.71-1.29-1.018-1.946.303-.657.646-1.313 1.013-1.954.38-.66.773-1.286 1.18-1.868.728-.064 1.466-.098 2.21-.098zm-3.635.254c-.24.377-.48.763-.704 1.16-.225.39-.435.782-.635 1.174-.265-.656-.49-1.31-.676-1.947.64-.15 1.315-.283 2.015-.386zm7.26 0c.695.103 1.365.23 2.006.387-.18.632-.405 1.282-.66 1.933-.2-.39-.41-.783-.64-1.174-.225-.392-.465-.774-.705-1.146zm3.063.675c.484.15.944.317 1.375.498 1.732.74 2.852 1.708 2.852 2.476-.005.768-1.125 1.74-2.857 2.475-.42.18-.88.342-1.355.493-.28-.958-.646-1.956-1.1-2.98.45-1.017.81-2.01 1.085-2.964zm-13.395.004c.278.96.645 1.957 1.1 2.98-.45 1.017-.812 2.01-1.086 2.964-.484-.15-.944-.318-1.37-.5-1.732-.737-2.852-1.706-2.852-2.474 0-.768 1.12-1.742 2.852-2.476.42-.18.88-.342 1.356-.494zm11.678 4.28c.265.657.49 1.312.676 1.948-.64.157-1.316.29-2.016.39.24-.375.48-.762.705-1.158.225-.39.435-.788.636-1.18zm-9.945.02c.2.392.41.783.64 1.175.23.39.465.772.705 1.143-.695-.102-1.365-.23-2.006-.386.18-.63.406-1.282.66-1.933zM17.92 16.32c.112.493.2.968.254 1.423.23 1.868-.054 3.32-.714 3.708-.147.09-.338.128-.563.128-1.012 0-2.514-.807-4.11-2.28.686-.72 1.37-1.536 2.02-2.44 1.107-.118 2.154-.3 3.113-.54zm-11.83.01c.96.234 2.006.415 3.107.532.66.905 1.345 1.727 2.035 2.446-1.595 1.483-3.092 2.295-4.11 2.295-.22-.005-.406-.05-.553-.132-.666-.38-.955-1.834-.73-3.703.054-.46.142-.944.25-1.438zm4.56.64c.44.02.89.034 1.345.034.46 0 .915-.01 1.36-.034-.44.572-.895 1.095-1.345 1.565-.455-.47-.91-.993-1.36-1.565z"
                  />
                </svg>
              </div>
              <div className="mt-5">
                <h2 className="text-lg font-medium text-gray-700">
                  What is missing from current LLMs?
                </h2>
                <p className="mt-2 text-base text-gray-500">
                  LLMs act as passive responders, especially when faced with ambiguous inputs. They don't naturally help users explore their needs in multiturn interations or offer suggestions for next steps.
                </p>
              </div>
            </div>

            <div className="mt-10 lg:mt-0">
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-orange-500 text-white">
                <Database className="h-6 w-6" />
              </div>
              <div className="mt-5">
                <h2 className="text-lg font-medium text-gray-700">
                  Why do LLMs fail to understand users?
                </h2>
                <p className="mt-2 text-base text-gray-500">
                Most LLMs are tuned based on single-turn human preferences. These single-turn rewards encourage models to generate response that may NOT be useful in the long term.
                </p>
              </div>
            </div>

            <div className="mt-10 lg:mt-0">
              <div className="flex items-center justify-center h-12 w-12 rounded-md bg-orange-500 text-white">
                <CreditCard className="h-6 w-6" />
              </div>
              <div className="mt-5">
                <h2 className="text-lg font-medium text-gray-700">
                  How do we build collaborative LLMs?
                </h2>
                <p className="mt-2 text-base text-gray-500">
                CollabLLM rewards LLMs responses based on their long-term impact on the conversation. By finetune LLMs using these long-term, interaction-level rewards, they actively seek information and collaborate more effectively with users.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-8 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-left">
            <h2 className="text-3xl font-bold text-gray-700 tracking-tight sm:text-4xl mb-8">
              What Users Said
              <span className=" text-orange-500"> About CollabLLM</span>
            </h2>
            
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-6xl mx-auto">
              {/* Quote 1 */}
              <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <div className="text-orange-500 text-2xl mb-4">&ldquo;Efficient</div>
                <p className="text-gray-600 italic mb-4">
                I was surprised by the first response. I was expecting a quick summary related to my prompt, but instead the AI asked me some questions. 
                I think this style worked well. 
                {/* It really helped me get detailed writing from the start.  */}
                {/* The response after that was more nuanced.  */}
                I felt like I had to do <strong>less editing</strong> to personalize the review.
                </p>
              </div>
              {/* Quote 2 */}
              <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <div className="text-orange-500 text-2xl mb-4">&ldquo;Stimulate Creativity</div>
                <p className="text-gray-600 italic mb-4">
                Asking questions and making you think of things you <strong>never thought of</strong>.
                </p>

                {/* <p className="text-gray-600 italic mb-4">
                Had some <strong>interesting ideas</strong> and asked good questions.
                </p> */}

                <p className="text-gray-600 italic mb-4">
                The AI assistant listened extremely well and offered suggestions that made sense as if it were a <strong>real conversation</strong>
                </p>                
              </div>

              {/* Quote 3 */}
              <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
                <div className="text-orange-500 text-2xl mb-4">&ldquo;Safer</div>
                <p className="text-gray-600 italic mb-4">
                The AI assistant told me why it <strong>wouldn't be helpful</strong> for this case.
                </p>
                <p className="text-gray-600 italic mb-4">
                It helped really well to navigate what to say and <strong>what information is needed</strong>.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="py-12 bg-gray-50 w-full">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:grid lg:grid-cols-2 lg:gap-8 lg:items-center">
            <div>
              <h2 className="text-3xl font-bold text-gray-700 sm:text-4xl">
                Ready to Make Your LLMs <span className=" text-orange-500"> Collaborative?</span>
              </h2>
              <p className="mt-3 max-w-3xl text-lg text-gray-500">
                Our code makes it easy for you to get more collaborative LLMs on your own tasks. 
                Don't waste time interacting with LLMs that fail to understand your need, start building collaborative LLMs!
              </p>
            </div>
            <div className="mt-8 lg:mt-0 flex justify-center lg:justify-end">
              <a href="https://github.com/Wuyxin/collabllm.git" target="_blank">
                <Button
                  size="lg"
                  variant="outline"
                  className="text-lg rounded-full"
                >
                  View code on Github
                  <ArrowRight className="ml-3 h-6 w-6" />
                </Button>
              </a>
            </div>
          </div>
        </div>
      </section>

{/* Blog Section */}
<section id="blog" className="py-16 bg-white">
  <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
    {/* Blog Header */}
    <div className="text-center mb-12">
      <h2 className="text-3xl font-bold text-gray-700 sm:text-4xl mb-3">
        From the Blog
      </h2>
      <p className="text-lg text-gray-500">
        Insights and updates from our research team
      </p>
    </div>

    {/* Blog Post */}
    <article className="max-w-none">
      {/* Blog Post Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-700 sm:text-4xl mb-4 font-serif">
          Building the Future of Collaborative AI: Our Journey with CollabLLM
        </h1>
        <div className="flex items-center gap-6 text-gray-500">
          <div className="flex items-center gap-2">
            <Calendar size={16} />
            <span className="text-sm">June 12, 2025</span>
          </div>
          <div className="flex items-center gap-2">
            <Clock size={16} />
            <span className="text-sm">6 min read</span>
          </div>
          <div className="flex items-center gap-2">
            <User size={16} />
            <span className="text-sm">Shirley Wu, Michel Galley</span>
          </div>
        <ViewerSystem />
          
        </div>
        
        {/* Viewer System Component */}
      </div>

      {/* Blog Content */}
      <div className="">
        <div className="prose prose-lg max-w-none font-serif leading-relaxed text-gray-500">
          <p className="text-xl text-gray-500 mb-6 italic">
            "The future of AI isn't just about making models smarter—it's about making them truly collaborative partners in human endeavors."
          </p>

          <h2 className="text-2xl font-bold text-gray-700 mt-10 mb-4">
            The Challenge We Set Out to Solve
          </h2>
          
              
        <p className="mb-4 text-lg">
          When we first started working with large language models, we noticed something puzzling. We saw that these models were incredibly capable. However, we all experienced a particular kind of frustration, illustrated perfectly by this example from{' '}
          <a 
            href="https://www.platformer.news/openai-operator-ai-agent-hands-on/" 
            className="text-blue-800 hover:text-blue-800 underline"
            target="_blank" 
            rel="noopener noreferrer"
          >
            Casey Newton
          </a>:
        </p>


          {/* Quote Box */}
          <blockquote className="border-l-4 border-gray-500 pl-6 py-4 mb-6 bg-gray-50 rounded-r-lg italic">
            <p className="text-lg text-gray-500 mb-3">
              My most frustrating experience with Operator was my first one: trying to order groceries. </p>
            <p className="text-lg text-gray-500 mb-3">
              <em>“Help me buy groceries on Instacart,”</em> I said, expecting it to ask me some basic questions:
              Where do I live? What store do I usually buy groceries from? What kinds of groceries do I want?
            </p>
            <p className="text-lg text-gray-500">
              It didn’t ask me any of that. Instead, Operator opened Instacart in a browser tab and began 
              searching for milk in grocery stores located in Des Moines, Iowa.
            </p>
          </blockquote>

          <p className="mb-4 text-lg text-gray-500">
            It’s genuinely surprising: one of the <strong>smartest LLMs</strong>—capable of solving graduate-level math problems—
            can still fail at basic human communication.
          </p>

          <p className="mb-4 text-lg text-gray-500">
            <strong>This is not a minor flaw.</strong> LLMs that lack effective communication skills pose challenges across key dimensions:
            <span className="italic"> performance, safety, and efficiency</span>. 
            Ask yourself:
          </p>

          <ul className="list-disc list-inside mb-4 text-lg text-gray-500">
            <li>How can we get satisfactory results if LLMs make assumptions about our preferences?</li>
            <li>How reliable is it to consult AI on healthcare, legal, or financial decisions?</li>
            <li>How much time and patience are we expected to waste just trying to get our point across?</li>
          </ul>

          <p className="mb-6 text-lg text-gray-500">
            The problem runs deeper. We typically evaluate LLMs in <strong>simple, sanitized test environments</strong>—single-turn prompts with clear, unambiguous instructions. But is that how real communication works?
          </p>

          <p className="mb-6 text-lg text-gray-500">
            In real life, solving meaningful problems requires <strong>collaboration, iteration, and contextual awareness</strong>. Moreover, if humans and LLMs are going to tackle groundbreaking problems together, AI systems can't just passively respond to human requests—they need to actively <strong>stimulate human creativity</strong> and guide the collaborative process. 
          </p>
          <p className="mb-6 text-lg text-gray-500">
            That’s why we’re introducing <span className="font-semibold text-black-500">CollabLLM</span>: 
            a framework designed to unlock the potential of human-AI collaboration by enabling LLMs to act 
            as <em>active, collaborative partners</em> rather than passive responders.
          </p>
          <h2 className="text-2xl font-bold text-gray-700 mt-10 mb-4">
            Our Breakthrough Approach
          </h2>

          <p className="mb-4 text-lg">
            The core idea behind CollabLLM is simple: in a multi-turn interaction, what matters most is not how good a single response is—but how it affects the rest of the conversation.
          </p>

          <p className="mb-4 text-lg">
            Take this scene from{' '}
            <a 
              href="https://www.youtube.com/watch?v=7fbaP2YjJ40&t=245s" 
              className="text-blue-800 hover:text-blue-800 underline"
              target="_blank" 
              rel="noopener noreferrer"
            >
              <em>Friends</em> (4:05 in the YouTube clip)
            </a>{' '}
            <a 
              href="https://www.bilibili.com/video/BV1vJ4m1j7zF/?spm_id_from=333.337.search-card.all.click" 
              className="text-blue-800 hover:text-blue-800 underline"
              target="_blank" 
              rel="noopener noreferrer"
            >
              / (1:42 in the Bilibili clip)
            </a>
            : Rachel and Joey are talking about dating strategies. Rachel asks a seemingly simple question: 
            <em>"So, where'd you grow up?"</em> Joey immediately mocks her—<em>"That's your move?"</em>—implying the question is naive. 
            But a few turns later, his tone changes. He's genuinely impressed: <em>"Wow!"</em>—because the question led him to open up and connect. The key insight? <strong>What matters isn't how a response is judged in the moment, but how it shapes the entire conversation.</strong>
          </p>

          <p className="mb-4 text-lg">
            Now imagine a model that chooses to ask a clarifying question instead of giving a direct answer. Standard reinforcement learning from human feedback (RLHF) might penalize that—it didn't provide information right away. But if the question helps uncover useful context that improves the conversation downstream, shouldn't it be rewarded?
          </p>

          {/* Key Concept Highlight */}
          <div className="bg-blue-50 border-l-4 border-blue-400 p-6 my-6 rounded-r-lg">
            <p className="text-lg text-gray-800 mb-0">
              That's exactly what CollabLLM does. We define a new reward function that measures the <strong>causal effect</strong> of a model's response on the future trajectory of a conversation. We call this the <strong>Multiturn-aware Reward (MR)</strong>. It evaluates a single model action based on its longer-term impact—not just immediate helpfulness.
            </p>
          </div>

          <p className="mb-4 text-lg">
            <strong>Quiz:</strong> is asking a question always better than giving an answer? The answer is—not necessarily. It depends entirely on the objective. 
            In most real-world situations, repeatedly asking questions without making progress is inefficient, because the ultimate goal remains unmet. 
            But take the game <em>20 Questions</em> as an example—where the objective is to guess what someone is thinking by asking a limited number of yes/no questions. 
            In that case, asking questions is essential, and giving an answer too early would break the format and defeat the purpose of the game. 
            This is where Multiturn-aware Reward (MR) comes in: it allows the model to adapt its behavior based on the context, learning <em>when</em> to ask and <em>when</em> to answer—depending entirely on what the task requires.
          </p>

          <p className="mb-4 text-lg">
            Now, going back to the <em>Friends</em> example with Rachel and Joey—how do we measure the value of Rachel's question over the course of a conversation? We need two components:<br />
            1) A <strong>user simulator</strong> to generate realistic follow-up responses (e.g., what Joey might say next), and<br />
            2) An <strong>evaluator</strong> to judge whether the interaction is successful—such as whether Joey becomes more romantically engaged.
          </p>

          <p className="mb-4 text-lg">
            Fortunately, both parts are quite feasible. First, the model you're training—let's call it "Rachel"—serves as the policy model generating responses. To simulate realistic dialogue, we prompt another model to act as "Joey," a proxy for the user. While inspired by our earlier example, "Joey" can represent <strong>any user simulator</strong>: a shopper trying to order groceries, a student asking math questions, or a writer seeking feedback. Second, we define task-specific metrics to evaluate success. In the dating example, it might be emotional engagement; in writing, it could be clarity or persuasiveness; in a question-answering task, it might be factual correctness. These evaluation criteria can even be combined—it's entirely up to your application!
          </p>

          <p className="mb-4 text-lg">
            With Multiturn-aware Reward in place, the goal becomes straightforward: train the policy model to maximize this reward. In doing so, the model learns to drive the conversation effectively toward the desired outcome—whether that's solving a task, clarifying a request, or building rapport.
          </p>
                    
        {/* Closing Statement */}
        <div className="bg-green-50 border-l-4 border-green-600 p-4 my-6 rounded-r-lg">
          <p className="text-lg text-gray-800 mb-0">
            After all, you don't need massive changes to build a collaborative model. Just a new way to define the objective—and a longer lens for measuring what matters in a conversation.
          </p>
        </div>


          <h2 className="text-2xl font-bold text-gray-700 mt-10 mb-4">
            Real-World Impact
          </h2>

          <p className="mb-4 text-gray-500 text-lg">
            The applications of collaborative AI are vast and exciting. From working on document editing to solving complex scientific problems, CollabLLM opens up new possibilities for human-AI collaboration.
          </p>

          <p className="mb-6 text-gray-500 text-lg">
            We've seen remarkable results in our initial testing, with collaborative LLMs outperforming non-collaboratively trained LLMs across various benchmarks. More importantly, users report a more efficient, engaging, and reliable interaction experience when working with the collaborative LLMs.
          </p>

          <h2 className="text-2xl font-bold text-gray-700 mt-10 mb-4">
            What's Next?
          </h2>

          <p className="mb-4 text-lg">
            We're continuously refining our approach, exploring new collaboration patterns. Our goal is to democratize collaborative AI and enable anyone to build more effective AI-powered solutions.
          </p>

          <p className="text-lg font-medium text-gray-500 mb-8">
            Join us in building the future of collaborative AI. Check out our code, contribute to the project, and help us shape the next generation of AI systems that truly understand the power of working together.
          </p>
        </div>

        {/* Call to Action */}
        <div className="mt-8 pt-6 border-t border-gray-200">
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <a href="https://github.com/Wuyxin/collabllm.git" target="_blank">
              <Button size="lg" className="rounded-full">
                Explore the Code
                <ArrowRight className="ml-2 h-5 w-5" />
              </Button>
            </a>
            <a href="https://arxiv.org/pdf/2502.00640" target="_blank">
              <Button size="lg" variant="outline" className="rounded-full">
                Read Our Paper
              </Button>
            </a>
          </div>
        </div>
      </div>
    </article>
  </div>
</section>
    </main>
  );
}