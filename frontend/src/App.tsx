import { Switch, Route, Router as WouterRouter, useLocation } from "wouter";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Suspense } from "react";
import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import Home from "@/pages/Home";
import About from "@/pages/About";
import Contact from "@/pages/Contact";
import SignIn from "@/pages/SignIn";
import Register from "@/pages/Register";
import DiseaseGuide from "@/pages/DiseaseGuide";
// @ts-ignore
import AppShell from "@/pages/AppShell";
// @ts-ignore
import { AuthProvider, useAuth } from "@/context/AuthContext";
// @ts-ignore
import { ToastProvider } from "@/components/Toast";

const queryClient = new QueryClient();

const publicPages = ["/", "/about", "/contact", "/diseases"];

function isPublicPage(path: string) {
  return publicPages.includes(path);
}

function Layout({ children, path }: { children: React.ReactNode; path: string }) {
  const showLayout = isPublicPage(path);
  return (
    <>
      {showLayout && <Navbar />}
      <main>{children}</main>
      {showLayout && <Footer />}
    </>
  );
}

function AuthRedirect({ children }: { children: React.ReactNode }) {
  const { user } = useAuth();
  const [, setLocation] = useLocation();
  if (user) {
    setLocation("/app");
    return null;
  }
  return <>{children}</>;
}

function Router() {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-green-50">
        <div className="flex flex-col items-center gap-4">
          <div className="w-10 h-10 border-4 border-green-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-green-700 text-sm font-medium">Loading CropGuard AI…</p>
        </div>
      </div>
    );
  }

  if (user) {
    return <AppShell />;
  }

  return (
    <Switch>
      <Route path="/">
        {() => <Layout path="/"><Home /></Layout>}
      </Route>
      <Route path="/about">
        {() => <Layout path="/about"><About /></Layout>}
      </Route>
      <Route path="/contact">
        {() => <Layout path="/contact"><Contact /></Layout>}
      </Route>
      <Route path="/diseases">
        {() => <Layout path="/diseases"><DiseaseGuide /></Layout>}
      </Route>
      <Route path="/signin">
        {() => <SignIn />}
      </Route>
      <Route path="/register">
        {() => <Register />}
      </Route>
      <Route path="/app">
        {() => <AppShell />}
      </Route>
      <Route>
        {() => (
          <Layout path="/">
            <div className="min-h-screen flex items-center justify-center">
              <div className="text-center">
                <h1 className="text-4xl font-bold text-gray-900 mb-4">404</h1>
                <p className="text-gray-500">Page not found</p>
              </div>
            </div>
          </Layout>
        )}
      </Route>
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <ToastProvider>
          <AuthProvider>
            <WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, "")}>
              <Suspense fallback={null}>
                <Router />
              </Suspense>
            </WouterRouter>
            <Toaster />
          </AuthProvider>
        </ToastProvider>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
