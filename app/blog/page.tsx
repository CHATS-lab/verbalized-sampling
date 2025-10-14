'use client';

import { useEffect } from 'react';

export default function BlogRedirect() {
  useEffect(() => {
    window.location.href = 'https://simonucl.notion.site/verbalized-sampling';
  }, []);

  return (
    <div style={{ padding: '20px', textAlign: 'center' }}>
      <p>Redirecting to blog...</p>
      <p>
        If you are not redirected automatically,{' '}
        <a href="https://simonucl.notion.site/verbalized-sampling">click here</a>.
      </p>
    </div>
  );
}
