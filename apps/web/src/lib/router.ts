import { useEffect, useState } from 'react';

export type Route =
  | { page: 'home' }
  | { page: 'job'; id: string }
  | { page: 'scene'; id: string }
  | { page: 'capture' }
  | { page: 'sandbox' };

function parse(): Route {
  if (location.pathname.startsWith('/capture')) return { page: 'capture' };
  const h = location.hash.replace(/^#\/?/, '');
  const [page, id] = h.split('/');
  if (page === 'job' && id) return { page: 'job', id };
  if (page === 'scene' && id) return { page: 'scene', id };
  if (page === 'sandbox') return { page: 'sandbox' };
  return { page: 'home' };
}

export function navigate(to: string): void {
  if (to.startsWith('#')) location.hash = to;
  else location.assign(to);
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(parse);
  useEffect(() => {
    const on = () => setRoute(parse());
    window.addEventListener('hashchange', on);
    window.addEventListener('popstate', on);
    return () => {
      window.removeEventListener('hashchange', on);
      window.removeEventListener('popstate', on);
    };
  }, []);
  return route;
}
