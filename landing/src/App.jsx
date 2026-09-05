import TitleBar from "./components/TitleBar.jsx";
import Hero from "./components/Hero.jsx";
import Pipeline from "./components/Pipeline.jsx";
import ProofSection from "./components/ProofSection.jsx";
import Caveats from "./components/Caveats.jsx";
import Footer from "./components/Footer.jsx";

export default function App() {
  return (
    <>
      <TitleBar />
      <main className="wrap">
        <Hero />
        <Pipeline />
        <ProofSection />
        <Caveats />
      </main>
      <Footer />
    </>
  );
}
