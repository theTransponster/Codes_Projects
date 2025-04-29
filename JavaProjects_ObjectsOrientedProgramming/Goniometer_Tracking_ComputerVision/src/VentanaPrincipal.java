import javax.swing.JFrame;

public class VentanaPrincipal extends JFrame{
	
	public VentanaPrincipal(){
		super("Goniometro Visual");
		this.setDefaultCloseOperation(EXIT_ON_CLOSE);
		PanelPrincipal p=new PanelPrincipal(this);
		this.setLocation(410, 130);
		this.add(p);
		this.pack();
		this.setVisible(true);
	}
	public static void main(String[] args) {
		VentanaPrincipal a=new VentanaPrincipal();
	}
}
